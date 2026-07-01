class TrainLogger():
    def __init__(self):
        self.data_loss = []
        self.initial_loss = []
        self.pde_loss = []
        self.ode_loss = []
        self.bc_loss = []
        self.total_loss = []
        self.F_loss = []
        self.loss_weights = []  # Track dynamic loss weights
        self.grad_norms = []  # Track gradient norms
        self.epoch_durations = []  # Track epoch durations


class ValLogger():
    def __init__(self):
        self.max_total_loss = []
        self.mean_total_loss = []
        self.data_loss = []
        self.init_loss = []
        self.pde_loss = []
        self.ode_loss = []
        self.bc_loss = []
        self.F_loss = []
        self.max_l2_loss = []
        self.mean_l2_loss = []