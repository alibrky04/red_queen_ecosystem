# An ecosystem to test enviromental constraints on a Red Queen relationship

from src.core.app_manager import AppManager
import src.configs.sim_config as sim_config
import pygame

if __name__ == "__main__":
    app = AppManager(
        prey_config_path='configs/prey-config.ini',
        predator_config_path='configs/predator-config.ini',
        width=sim_config.WIDTH,
        height=sim_config.HEIGHT,
        fps=sim_config.FPS,
        render_every_n=sim_config.RENDER_N
    )

    next_action = "MAIN_MENU"

    while True:
        if app.app_quit_requested:
            break

        if next_action == "MAIN_MENU":
            choice = app.show_main_menu()

            if choice == "START":
                app.reset_training_state(clear_metrics=True)
                run_status = app.start_training(generations=sim_config.GENERATIONS)

                if run_status == "EXIT" or app.app_quit_requested:
                    break

                title = "Training Finished" if run_status == "COMPLETED" else "Training Stopped"
                next_action = app.show_end_screen(title)
            
            elif choice == "BATCH":
                app.run_batch_evaluation(
                    seeds=[0, 1, 2, 3, 4],
                    generations=sim_config.GENERATIONS,
                    output_dir="results/batch_current_config"
                )

                if app.app_quit_requested:
                    break

                next_action = app.show_end_screen("Batch Evaluation Finished")

            elif choice == "LOAD":
                prey_genome, pred_genome = app.load_best_models(use_dialog=True)
                if prey_genome and pred_genome:
                    app.watch_best_models(prey_genome, pred_genome, rounds=1)

                    if app.app_quit_requested:
                        break

                    next_action = app.show_end_screen("Loaded Model Finished")
                else:
                    next_action = app.show_end_screen("No Saved Model Found")

            else:
                break

        elif next_action == "NEW_RUN":
            app.reset_training_state(clear_metrics=True)
            run_status = app.start_training(generations=sim_config.GENERATIONS)

            if run_status == "EXIT" or app.app_quit_requested:
                break

            title = "Training Finished" if run_status == "COMPLETED" else "Training Stopped"
            next_action = app.show_end_screen(title)

        elif next_action == "LOAD_LAST":
            prey_genome, pred_genome = app.load_best_models(use_dialog=True)
            if prey_genome and pred_genome:
                app.watch_best_models(prey_genome, pred_genome, rounds=1)

                if app.app_quit_requested:
                    break

                next_action = app.show_end_screen("Loaded Model Finished")
            else:
                next_action = app.show_end_screen("No Saved Model Found")

        elif next_action == "EXIT":
            break

        else:
            next_action = "MAIN_MENU"

    if app.render_enabled and app.renderer:
        app.renderer.quit()
    else:
        pygame.quit()