import random
import copy
import numpy as np
import json
    
def select_demonstration(support_meta, n_shot, dataset, query=None, selection_strategy="random"):
    if 'pendulum' in dataset or 'flow' in dataset or 'circuit' in dataset:
        # operator_index = {'+': 0, '-': 1, 'x': 2}
        
        if selection_strategy == "random":
            n_shot_support_raw = random.sample(support_meta, n_shot)
            n_shot_support = copy.deepcopy(n_shot_support_raw)
        else:
            
            if dataset == 'pendulum':
                with open('./new_data/intervention/pendulum/angle.json', 'r') as f:
                    support_zero = json.load(f)
                    
                with open('./new_data/intervention/pendulum/light.json', 'r') as f:
                    support_one = json.load(f)
                    
                with open('./new_data/intervention/pendulum/shadow_len.json', 'r') as f:
                    support_two = json.load(f)
                    
                with open('./new_data/intervention/pendulum/shadow_pos.json', 'r') as f:
                    support_three = json.load(f)

                if n_shot == 4:
                    n_shot_support = random.sample(support_zero, 1) + random.sample(support_one, 1) + random.sample(support_two, 1) + random.sample(support_three, 1) + random.sample(support_zero, 1) + random.sample(support_one, 1)
                elif n_shot == 8:
                    n_shot_support = random.sample(support_zero, 2) + random.sample(support_one, 2) + random.sample(support_two, 2) + random.sample(support_three, 2)

                return n_shot_support
            elif dataset == 'flow':
                
                with open('./new_data/intervention/flow/ball_size.json', 'r') as f:
                    support_zero = json.load(f)
                
                with open('./new_data/intervention/flow/hole_position.json', 'r') as f:
                    support_one = json.load(f)

                with open('./new_data/intervention/flow/water_level.json', 'r') as f:
                    support_two = json.load(f)
                
                if n_shot == 4:
                    n_shot_support = random.sample(support_zero, 1) + random.sample(support_one, 1) + random.sample(support_two, 1)
                elif n_shot == 8:
                    n_shot_support = random.sample(support_zero, 2) + random.sample(support_one, 2) + random.sample(support_two, 2)
                
                return n_shot_support
            
            elif dataset == 'circuit':
                
                with open('./new_data/intervention/circuit/red_light.json', 'r') as f:
                    support_zero = json.load(f)
                    
                with open('./new_data/intervention/circuit/green_light.json', 'r') as f:
                    support_one = json.load(f)
                
                with open('./new_data/intervention/circuit/blue_light.json', 'r') as f:
                    support_two = json.load(f)
                
                with open('./new_data/intervention/circuit/robot_arm.json', 'r') as f:
                    support_three = json.load(f)
                
                if n_shot == 4:
                    n_shot_support = random.sample(support_zero, 1) + random.sample(support_one, 1) + random.sample(support_two, 1) + random.sample(support_three, 1)
                elif n_shot == 8:
                    n_shot_support = random.sample(support_zero, 2) + random.sample(support_one, 2) + random.sample(support_two, 2) + random.sample(support_three, 2)

                return n_shot_support
                            
            # # interv = int(support_meta['image'][1].split('_')[-1].removesuffix('.png')[1:-1])
        
            # supp_size = len(support_meta)
            
            # supp_indices = np.arange(0, supp_size, 3, dtype=int)
            
            # n_shot_support_raw = random.sample(supp_indices.tolist(), 1)[0]
            
            # n_shot_support = [support_meta[n_shot_support_raw], support_meta[n_shot_support_raw + 1], support_meta[n_shot_support_raw + 2]]
            
            # n_shot_support = sorted(n_shot_support, key=lambda d: d['answer'])
            # n_shot_support = n_shot_support[1:]
            # print(sorted_support)
            # exit(0)
            # n_shot_support = copy.deepcopy(n_shot_support_raw)
            # operator = query['operator']
            # operator_idx = operator_index[operator]
            # for support in n_shot_support:    
            #     support['answer'] = support['answer'][operator_idx]

def get_task_instruction(args):
    dataset = args.dataset
    description = args.task_description
    if description == 'nothing':
        instr = ''
        return instr
    
    if dataset == "circuit":
        if description == 'intervention':            
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You are given two images of a robotic scene: a before image and an after image. Each image shows a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables: robot arm, green light, blue light, and red light. These variables are causally related as follows:\n(1) Changing the arm position causes one button to be pressed, which directly affects the corresponding light (red, green, or blue).\n(2) Turning on the green or blue light can indirectly activate the Red light.\n(3) Changing any light does not affect the arm position.\nYour task is to compare the two images, identify the first variable that changed, and use the causal rules above to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: robot arm, green light, blue light, red light."
        elif description == 'intervention_nograph':            
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You are given two images of a robotic scene: a before image and an after image. Each image shows a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables: robot arm, green light, blue light, and red light. These variables are causally related.\nYour task is to compare the two images, identify the first variable that changed, and to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: robot arm, green light, blue light, red light."
        elif description == 'counterfactual':            
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup showing a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables: robot arm, green light, blue light, and red light. The robot arm can be one of the following values: touching red light, touching blue light, touching green light, or not touching any light. The red light can be one of the following values: on or off. The green light can be one of the following values: on or off. The blue light can be one of the following values: on or off. These variables are causally related as follows:\n(1) Changing the arm position causes one button to be pressed, which directly affects the corresponding light (red, green, or blue).\n(2) Turning on the green or blue light can indirectly activate the Red light.\n(3) Changing any light does not affect the arm position.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
        elif description == 'counterfactual_nograph':            
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup showing a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables: robot arm, green light, blue light, and red light. The robot arm can be one of the following values: touching red light, touching blue light, touching green light, or not touching any light. The red light can be one of the following values: on or off. The green light can be one of the following values: on or off. The blue light can be one of the following values: on or off. These variables are causally related.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
        elif description == 'discovery':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup showing a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables that are causally related: robot arm, green light, blue light, and red light. Given an image and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."   
        elif description == 'discovery_interleaved':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You are given two images of a robotic scene. The first image shows a robotic arm positioned over a circular arc with three buttons (red, green, blue) and the resulting lighting in the scene. The scene contains four variables that are causally related: robot arm, green light, blue light, and red light. The second image shows the same setup after one of these variables is initially changed and other variables may have changed as a downstream effect. Given a pair of images and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."    
    elif dataset == "flow":
        if description == 'intervention':   
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images. The first image shows a physical setup with water in a glass and a hole on the right side of the glass from where the water is flowing. There is also a red ball inside the glass that affects the water level in the glass and the water flow from the hole. The second image shows the same setup after a change has occurred. The scene contains four variables: ball size, water level, hole position, and water flow. These variables are causally related as follows:\n(1) If the ball size changes, it causes the water level to change and affects water flow. It does NOT cause hole position to change.\n(2) If the water level changes, it causes the water flow to change. It does NOT cause ball size and hole position to change.\n(3) If the hole position changes, it causes water flow to change. It does NOT cause ball size or water level to change.\nYour task is to compare the two images, identify the first variable that changed, and use the causal rules above to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: ball size, water level, hole position."
        elif description == 'intervention_nograph':   
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images. The first image shows a physical setup with water in a glass and a hole on the right side of the glass from where the water is flowing. There is also a red ball inside the glass that affects the water level in the glass and the water flow from the hole. The second image shows the same setup after a change has occurred. The scene contains four variables: ball size, water level, hole position, and water flow. These variables are causally related.\nYour task is to compare the two images, identify the first variable that changed, and to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: ball size, water level, hole position."
        elif description == 'counterfactual':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with water in a glass and a hole on the right side of the glass from where the water is flowing. There is also a red ball inside the glass that affects the water level in the glass and the water flow from the hole. The scene contains four variables: ball size, water level, hole position, and water flow. The ball size can be one of the following values: small, medium, large. The hole position can be one of the following values: bottom, middle, top. The water level can be one of the following values: low, medium, high. The water flow can be one of the following values: left, middle, right. For water level, left refers to close to the glass and right refers to far from the glass. These variables are causally related as follows:\n(1) If the ball size changes, it causes the water level to change and affects water flow. It does NOT cause hole position to change.\n(2) If the water level changes, it causes the water flow to change. It does NOT cause ball size and hole position to change.\n(3) If the hole position changes, it causes water flow to change. It does NOT cause ball size or water level to change. \nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
        elif description == 'counterfactual_nograph':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with water in a glass and a hole on the right side of the glass from where the water is flowing. There is also a red ball inside the glass that affects the water level in the glass and the water flow from the hole. The scene contains four variables: ball size, water level, hole position, and water flow. The ball size can be one of the following values: small, medium, large. The hole position can be one of the following values: bottom, middle, top. The water level can be one of the following values: low, medium, high. The water flow can be one of the following values: left, middle, right. For water level, left refers to close to the glass and right refers to far from the glass. These variables are causally related.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
        elif description == 'discovery':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with water in a glass and a hole on the right side of the glass from where the water is flowing. There is also a red ball inside the glass that affects the water level in the glass and the water flow from the hole. The scene contains four variables that are causally related: ball size, water level, hole position, and water flow. Given an image and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."   
        elif description == 'discovery_interleaved':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images: the first image shows a physical setup with water in a glass, a hole on the right side of the glass from where the water is flowing, and a red ball inside the glass. The scene contains four variables that are causally related: ball size, hole position, water level, and water flow. The second image shows the same setup after one of these variables is initially changed and other variables may have changed as a downstream effect. Given a pair of images and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."     
    elif dataset == "pendulum":
        if description == 'intervention':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images: the first image shows a physical setup with a light source, a pendulum, and the pendulum’s shadow. The second image shows the same setup after a change has occurred. The scene contains four variables: pendulum angle, light position, shadow length, and shadow position. These variables are causally related as follows:\n(1) If the pendulum angle changes, it causes both the shadow length and shadow position to change. It does NOT cause the light position to change.\n(2) If the light position changes, it causes both the shadow length and shadow position to change. It does NOT cause the pendulum angle to change.\n(3) A change in shadow length does NOT cause any other variable to change.\n(4) A change in shadow position does NOT cause any other variable to change. \nYour task is to compare the two images, identify the first variable that changed, and use the causal rules above to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: pendulum angle, light position, shadow length, or shadow position."
        elif description == 'intervention_nograph':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images: the first image shows a physical setup with a light source, a pendulum, and the pendulum’s shadow. The second image shows the same setup after a change has occurred. The scene contains four variables: pendulum angle, light position, shadow length, and shadow position. These variables are causally related.\nYour task is to compare the two images, identify the first variable that changed, and to determine which variable is the likely root cause of any other changes. Respond with only one of the following variable names, exactly as written: pendulum angle, light position, shadow length, or shadow position."
        elif description == 'counterfactual': 
            # instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with a light source, a pendulum, and the pendulum’s shadow. The scene contains four variables: pendulum angle, light position, shadow length, and shadow position. The pendulum angle can be one of the following values: left, center, right. The light position can be one of the following values: right, center, left. The shadow length can be one of the following values: short, medium, long. The shadow position can be one of the following values: left, center, right. These variables are causally related as follows:\n(1) If the pendulum angle changes, it causes both the shadow length and shadow position to change. It does NOT cause the light position to change.\n(2) If the light position changes, it causes both the shadow length and shadow position to change. It does NOT cause the pendulum angle to change.\n(3) A change in shadow length does NOT cause any other variable to change.\n(4) A change in shadow position does NOT cause any other variable to change.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with a light source, a pendulum, and the pendulum’s shadow. The scene contains four variables: pendulum angle, light position, shadow length, and shadow position. The pendulum angle can be one of the following values: left, center, right. The light position can be one of the following values: right, center, left. The shadow length can be one of the following values: short, medium, long. The shadow position can be one of the following values: left, center, right. These variables are causally related as follows:\n(1) If the pendulum angle changes, it causes both the shadow length and shadow position to change. It does NOT cause the light position to change.\n(2) If the light position changes, it causes both the shadow length and shadow position to change. It does NOT cause the pendulum angle to change.\n(3) A change in shadow length does NOT cause any other variable to change.\n(4) A change in shadow position does NOT cause any other variable to change.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables would be had the variable been changed to the specified value."
        elif description == 'counterfactual_nograph': 
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with a light source, a pendulum, and the pendulum’s shadow. The scene contains four variables: pendulum angle, light position, shadow length, and shadow position. The pendulum angle can be one of the following values: left, center, right. The light position can be one of the following values: right, center, left. The shadow length can be one of the following values: short, medium, long. The shadow position can be one of the following values: left, center, right. These variables are causally related.\nGiven an image and a variable that will change, your task is to determine what the final values of all four variables will be after the change."
        elif description == 'discovery':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown an image containing a physical setup with a light source, a pendulum, and the pendulum’s shadow. The scene contains four variables that are causally related: pendulum angle, light position, shadow length, and shadow position. Given an image and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."   
        elif description == 'discovery_interleaved':
            instr = "You are a highly capable AI system specialized in causal reasoning from visual data. You will be shown two images: the first image shows a physical setup with a light source, a pendulum, and the pendulum’s shadow. The scene contains four variables that are causally related: pendulum angle, light position, shadow length, and shadow position. The second image shows the same setup after one of these variables is initially changed and other variables may have changed as a downstream effect. Given a pair of images and a question about two variables, A and B, your task is to determine whether A causes B. Answer simply with Yes or No."     

    
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['pendulum', 'flow', 'circuit']:
        answer = str(answer)
    return answer
