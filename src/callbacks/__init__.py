from src.callbacks.vector import register_vector_callbacks
from src.callbacks.matrix import register_matrix_callbacks
from src.callbacks.ui import register_ui_callbacks
from src.callbacks.animation import register_animation_callbacks

def register_all_callbacks(app_instance):
    register_vector_callbacks(app_instance)
    register_matrix_callbacks(app_instance)
    register_ui_callbacks(app_instance)
    register_animation_callbacks(app_instance)

