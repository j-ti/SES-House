import os
import fnmatch

from sklearn.metrics import mean_squared_error

from forecast import load_model, buildSet
from forecast_load import getNormalizedParts
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


model_paths = find("model_ts30*.json", "./output/forecast/load/")
print(model_paths)

best_model = ""
best_value = 1.0

for model_path in model_paths:
    model_path_h5 = model_path.replace("json", "h5")
    model = load_model(model_path, model_path_h5)

    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    loadConfig.LOOK_BACK = model.layers[0].input_shape[1]

    _, validation_part, _, scaler = getNormalizedParts(
        config, loadConfig, config.TIMESTAMPS
    )
    validation_x, validation_y = buildSet(
        validation_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE
    )

    validation_prediction = model.predict(validation_x)
    validation_mse = mean_squared_error(validation_y, validation_prediction)
    if validation_mse < best_value:
        best_value = validation_mse
        best_model = model_path
    print("validation mse: ", validation_mse)

print(best_model)
print(best_value)
