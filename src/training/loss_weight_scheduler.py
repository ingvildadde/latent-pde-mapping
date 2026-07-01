import torch
from typing import Dict, Optional, List
import numpy as np


class GradNormLossWeightScheduler:
    """
    Dynamically adapts loss weights based on gradient norms.
    
    Args:
        loss_names: List of loss component names (e.g., ['data', 'init', 'pde', 'ode', 'bc'])
        initial_weights: Initial weights for each loss component
        alpha: Restoring force rate for gradient norm adaptation (default: 1.5)
        update_frequency: How often to update weights (in training steps)
        adaptation_method: Method for computing weights ('grad_norm', 'equal', 'inverse_grad')
        warmup_steps: Number of steps before starting adaptation
        device: torch device
    """
    
    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Dict[str, float],
        alpha: float = 1.5,
        update_frequency: int = 1,
        adaptation_method: str = 'grad_norm',
        warmup_steps: int = 0,
        device: torch.device = torch.device('cuda')
    ):
        self.loss_names = loss_names
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.adaptation_method = adaptation_method
        self.warmup_steps = warmup_steps
        self.device = device
        
        # Initialize weights as learnable parameters
        self.weights = {name: initial_weights.get(name, 1.0) for name in loss_names}
        
        # Track initial loss values for normalization
        self.initial_losses: Optional[Dict[str, float]] = None
        
        # Track gradient norms history
        self.grad_norm_history: Dict[str, List[float]] = {name: [] for name in loss_names}
        
        # Track weight history
        self.weight_history: Dict[str, List[float]] = {name: [] for name in loss_names}
        
    def compute_gradient_norms(
        self,
        model: torch.nn.Module,
        losses: Dict[str, torch.Tensor],
        grad_norm_dict: Optional[Dict[str, float]],
        grad_clip_norm: Optional[float] = None,
        retain_graph: bool = True
    ) -> Dict[str, float]:
        """
        Compute gradient norms for each loss component.
        
        Args:
            model: The neural network model
            losses: Dictionary of loss values (not weighted)
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradient norms for each loss component
        """
        # grad_norms = {}
        
        for loss_name in grad_norm_dict.keys():
            if loss_name not in losses:
                grad_norm_dict[loss_name] = 0.0
                continue
                
            loss = losses[loss_name]
            
            # Zero out gradients
            model.zero_grad()
            
            # Compute gradients for this loss
            if loss.requires_grad:
                loss.backward(retain_graph=retain_graph)

                # Store gradients for L2 norm computation
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norm_dict[loss_name] += total_norm
            else:
                grad_norm_dict[loss_name] = 0.0

        # Zero gradients after computing norms
        model.zero_grad()

        return grad_norm_dict

    def update_weights(
        self,
        grad_norms: Dict[str, float],
        losses: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, float]:
        """
        Update loss weights based on gradient norms.
        
        Args:
            model: The neural network model
            losses: Dictionary of unweighted loss values
            
        Returns:
            Updated weights dictionary
        """
        
        # During warmup, use initial weights
        if epoch < self.warmup_steps:
            return self.weights.copy()
        
        # Only update at specified frequency
        if epoch % self.update_frequency != 0:
            return self.weights.copy()
        
        # Compute gradient norms
        # return self.compute_gradient_norms(model, losses, retain_graph=True)
        # grad_norms = self.compute_gradient_norms(model, losses, retain_graph=True)
        
        # Store gradient norms
        for name, norm in grad_norms.items():
            self.grad_norm_history[name].append(norm)
        
        # Update weights based on adaptation method
        if self.adaptation_method == 'grad_norm':
            self._update_weights_grad_norm(grad_norms, losses)
        elif self.adaptation_method == 'grad_norm_fixed_average':
            self._update_weights_grad_norm_fixed_average(grad_norms, epoch)
        elif self.adaptation_method == 'inverse_grad':
            self._update_weights_inverse_grad(grad_norms)
        elif self.adaptation_method == 'equal':
            self._update_weights_equal(grad_norms)
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
        
        # Store weight history
        for name, weight in self.weights.items():
            self.weight_history[name].append(weight)
        
        return self.weights.copy()

    
    def _update_weights_grad_norm_fixed_average(
        self,
        grad_norms: Dict[str, float],
        epoch: int
    ):
        """
        Update weights using GradNorm algorithm.
        
        Balances loss components based on their gradient magnitudes and
        relative training rates.
        """

        # During warmup, use initial weights
        if epoch < self.warmup_steps:
            return self.weights.copy()
        
        # Only update at specified frequency
        if epoch % self.update_frequency != 0:
            return self.weights.copy()

        # grad_norms = {k: v**0.5 for k, v in grad_norms.items() if v > 0}

        # Store gradient norms
        for name, norm in grad_norms.items():
            self.grad_norm_history[name].append(norm)

        updated_weights = {}
        for k in grad_norms.keys():
            norm_weight = sum(grad_norms.values()) / grad_norms[k]
            updated_weights[k] = (1-self.alpha)*self.weights[k] + self.alpha*norm_weight

        self.weights = updated_weights

        # Store weight history
        for name, weight in self.weights.items():
            self.weight_history[name].append(weight)
        
        return self.weights.copy()
    

    def _update_weights_grad_norm(
        self,
        grad_norms: Dict[str, float],
        epoch: int
    ):
        """
        Update weights using GradNorm algorithm.
        
        Balances loss components based on their gradient magnitudes and
        relative training rates.
        """

        # During warmup, use initial weights
        if epoch < self.warmup_steps:
            return self.weights.copy()
        
        # Only update at specified frequency
        if epoch % self.update_frequency != 0:
            return self.weights.copy()

        # grad_norms = {k: v**0.5 for k, v in grad_norms.items() if v > 0}

        # Store gradient norms
        for name, norm in grad_norms.items():
            self.grad_norm_history[name].append(norm)

        updated_weights = {}
        for k in grad_norms.keys():
            norm_weight = sum(grad_norms.values()) / grad_norms[k]
            updated_weights[k] = self.alpha*self.weights[k] + (1-self.alpha)*norm_weight

        self.weights = updated_weights

        # Store weight history
        for name, weight in self.weights.items():
            self.weight_history[name].append(weight)
        
        return self.weights.copy()

    
    def _update_weights_inverse_grad(self, grad_norms: Dict[str, float]):
        """
        Update weights inversely proportional to gradient norms.
        
        This gives higher weight to loss components with smaller gradients.
        """
        # Compute inverse of gradient norms
        inverse_norms = {
            name: 1.0 / (norm + 1e-8) for name, norm in grad_norms.items()
        }
        
        # Normalize to sum to number of losses
        total_inverse = sum(inverse_norms.values())
        num_losses = len(self.loss_names)
        
        for name in self.loss_names:
            self.weights[name] = (inverse_norms[name] / total_inverse) * num_losses
    
    def _update_weights_equal(self, grad_norms: Dict[str, float]):
        """
        Update weights to equalize gradient norms across all losses.
        """
        grad_norm_values = [v for v in grad_norms.values() if v > 0]
        if len(grad_norm_values) == 0:
            return
        
        mean_grad_norm = np.mean(grad_norm_values)
        
        for name in self.loss_names:
            if grad_norms[name] > 0:
                ratio = mean_grad_norm / (grad_norms[name] + 1e-8)
                self.weights[name] = self.weights[name] * (0.9 + 0.1 * ratio)  # Smooth update
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Get history of weight updates."""
        return self.weight_history.copy()
    
    def get_grad_norm_history(self) -> Dict[str, List[float]]:
        """Get history of gradient norms."""
        return self.grad_norm_history.copy()


class StaticLossWeightScheduler:
    """
    Simple scheduler that returns static weights.
    
    This is useful for maintaining compatibility when dynamic weighting is not used.
    """
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.weight_history = {name: [] for name in weights.keys()}
        self.grad_norm_history = {name: [] for name in weights.keys()}
    
    def update_weights(
        self,
        model: torch.nn.Module,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Return static weights without updates."""
        return self.weights.copy()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Get empty history."""
        return self.weight_history.copy()
    
    def get_grad_norm_history(self) -> Dict[str, List[float]]:
        """Get empty history."""
        return self.grad_norm_history.copy()
