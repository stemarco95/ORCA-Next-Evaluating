import copy
from typing import Dict

import torch
from core.base_module import BaseModule
from utils.context import Context

MODEL_KEY = "raw_model"
DISTURBANCE_KEY = "disturbance"

class SoftError(BaseModule):
    def __init__(
            self, 
            module_id, 
            inputs, 
            outputs, 
            cycle, 
            seed,
            is_env=False, 
            config=None
        ):
        super().__init__(module_id, inputs, outputs, cycle, seed, is_env, config)

        self.torch_rng = torch.Generator(device='cpu')
        self.torch_rng.manual_seed(self.local_seed)

        

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:

        disturbance_ctx = inputs.get(DISTURBANCE_KEY, None)    
        model_ctx = inputs.get(MODEL_KEY, None)     

        model = model_ctx.info.get("model")
        if model is None:
            raise ValueError("Model context must contain 'model' in info")

        target_parameter = disturbance_ctx.info.get("target_parameter")
        value = disturbance_ctx.info.get("value")
        sparsity = 0.2

        
        if target_parameter and value is not None:
            if target_parameter == "soft_error":
                std_dev = float(value) 
                
                with torch.no_grad():
                    for name, param in model.policy.actor.named_parameters():
                    
                        # 2. GENERATE BASE NOISE
                        noise = torch.normal(
                            mean=0.0, 
                            std=std_dev, 
                            size=param.size(), 
                            generator=self.torch_rng
                        ).to(param.device)
                        
                        # 3. APPLY SPARSITY MASK (if sparsity < 1.0)
                        if sparsity < 1.0:
                            # Create a boolean mask where `sparsity` % of elements are True
                            mask = torch.rand(
                                param.size(), 
                                generator=self.torch_rng
                            ).to(param.device) < sparsity
                            
                            # Zero out the noise where the mask is False
                            noise = noise * mask

                        # 4. INJECT NOISE
                        param.add_(noise)

        output_ctx = copy.copy(model_ctx)
        output_ctx.info["model"] = model  # Update the model in the context with the disturbed version, should happen in place

        return {
            self.outputs[0]: output_ctx
        }