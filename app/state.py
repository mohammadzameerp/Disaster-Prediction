class AppState:
    def __init__(self):
        self.df = None
        self.dataset_path = None
        self.summary = None
        self.artifacts = None
        self.model_bundle = None
        self.flood_models = None
        self.metrics = None
        self.current_algo = None
        self.metrics_map = {}
        self.home_bg = None
        self.about_imgs = []

state = AppState()
