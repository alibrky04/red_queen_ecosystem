# An ecosystem to test enviromental constraints on a Red Queen relationship

from src.core.app_manager import AppManager
import src.configs.sim_config as sim_config

if __name__ == "__main__":
    app = AppManager(
        prey_config_path='configs/prey-config.ini',
        predator_config_path='configs/predator-config.ini',
        width=sim_config.WIDTH,
        height=sim_config.HEIGHT,
        fps=sim_config.FPS,
        render_every_n=sim_config.RENDER_N
        )
    
    choice = app.show_main_menu()
    
    if choice == "START":
        app.start_training(generations=sim_config.GENERATIONS)
        
        best_prey = app.prey_pop.best_genome
        best_pred = app.predator_pop.best_genome
        if best_prey and best_pred:
            app.save_best_models(best_prey, best_pred)
            
    elif choice == "LOAD":
        prey_genome, pred_genome = app.load_best_models()
        if prey_genome and pred_genome:
            # TODO Code to evaluate a single generation using these two genomes visually
            pass