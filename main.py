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
    app.start_training(generations=sim_config.GENERATIONS)