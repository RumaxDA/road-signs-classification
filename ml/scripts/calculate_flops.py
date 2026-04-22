import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === MODELE ===
MODELS_TO_TEST = {
    "Custom CNN (32x32)": "/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_32/cnn_32_v1.keras",
    "Custom CNN (48x48)": "/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_48/cnn_48_v1.keras",
    "Custom CNN (96x96)": "/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_96/cnn_96_v1.keras",
    "Custom CNN (224x224)": "/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_224/cnn_224_v1.keras",
    "EfficientNet (32x32)": "/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_32/efficientnet_b0_32_v1.keras",
    "EfficientNet (48x48)": "/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_48/efficientnet_b0_48_v1.keras",
    "EfficientNet (96x96)": "/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_96/efficientnet_b0_96_v1.keras",
    "EfficientNet (224x224)": "/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_224/efficientnet_b0_224_v1.keras"
}

SAVE_DIR = '/home/rumaxx/road-signs-project/ml/new_plots/flops'
os.makedirs(SAVE_DIR, exist_ok=True)
REPORT_PATH = os.path.join(SAVE_DIR, "flops_report.txt")

def strip_problematic_layers(layer):
    """
    Agresywna funkcja klonująca. Wycinamy wszystko, co opiera się na losowości (RNG) 
    lub zachowuje się inaczej w trybie treningu (np. Dropout).
    """
    bad_layers = ['Random', 'Rescaling', 'Resizing', 'Dropout', 'SpatialDropout', 'GaussianNoise']
    if any(name in layer.__class__.__name__ for name in bad_layers):
        return tf.keras.layers.Activation('linear', name=layer.name + "_stripped")
    return layer

def get_flops(model):
    """
    Oblicza FLOPs dla modelu Keras, wymuszając tryb inferencji.
    """
    input_shape = (1,) + model.input_shape[1:]
    
    @tf.function
    def model_func(inputs):
        return model(inputs, training=False)

  
    concrete_func = model_func.get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        
        opts = tf.compat.v1.profiler.ProfileOptionBuilder(opts).with_empty_output().build()
        
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        
        if flops is None:
            return 0
            
        return flops.total_float_ops

if __name__ == "__main__":
    
    lines_to_print = []
    lines_to_print.append(f"{'='*65}")
    lines_to_print.append(f"{'MODEL':<25} | {'PARAMETRY':<15} | {'FLOPS'}")
    lines_to_print.append(f"{'-'*65}")

    for name, path in MODELS_TO_TEST.items():
        if not os.path.exists(path):
            lines_to_print.append(f"{name:<25} | {'BŁĄD: BRAK PLIKU':<15} | -")
            continue

        try:
            original_model = tf.keras.models.load_model(path, compile=False)
            clean_model = tf.keras.models.clone_model(original_model, clone_function=strip_problematic_layers)
            
            params = clean_model.count_params()
            flops = get_flops(clean_model)
            
            params_str = f"{params / 1e6:.2f} M" if params >= 1e6 else f"{params / 1e3:.2f} K"
            flops_str = f"{flops / 1e9:.3f} GFLOPs" if flops >= 1e9 else f"{flops / 1e6:.2f} MFLOPs"

            lines_to_print.append(f"{name:<25} | {params_str:<15} | {flops_str}")
            print(f"✓ Przeliczono: {name}") 
            
        except Exception as e:
            error_msg = f"{name:<25} | {'BŁĄD WYZNACZANIA':<15} | Patrz konsola"
            lines_to_print.append(error_msg)
            print(f"BŁĄD ({name}): {str(e)[:150]}...") 

    lines_to_print.append(f"{'='*65}")
    lines_to_print.append("M = Miliony, K = Tysiące")


    print("\n--- WYNIKI KOŃCOWE ---")
    for line in lines_to_print:
        print(line)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        for line in lines_to_print:
            f.write(line + "\n")
            
    print(f"\n✓ Zapisano raport do pliku: {REPORT_PATH}")