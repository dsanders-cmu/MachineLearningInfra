
class Model:
    
    def __init__(self, params, dim_in, dim_out, base_dir, model_dir, log_dir):
        # Save parameters
        self.params = params

        # Save input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Save directories
        self.base_dir = base_dir
        self.model_dir = model_dir
        self.log_dir = log_dir

    def predict(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass