from flask import Flask, render_template, jsonify, request
import numpy as np
from joblib import load
import os

# Cargar el modelo
dt = load("dt1.joblib")

# Generar el servidor (Back-End)
servidorWeb = Flask(__name__)


@servidorWeb.route("/holamundo", methods=["GET"])
def holamundo():
    return render_template("pagina1.html")


# Envío de datos a través de JSON
@servidorWeb.route("/modeloForm", methods=["POST"])
def modeloPrediccion():
    # Procesar los datos de entrada
    contenido = request.json
    print(contenido)
    datrosEntrada = np.array(
        [
            0.88,
            0,
            2.6,
            0.098,
            25,
            67,
            0.9968,
            1,
            0.4,
            contenido["pH"],
            contenido["sulphates"],
            contenido["alcohol"],
        ]
    )

    # Utilizar el modelo
    resultado = dt.predict(datrosEntrada.reshape(1, -1))
    print(datrosEntrada)
    return jsonify({"resultado": str(resultado[0])})


if __name__ == "__main__":
    servidorWeb.run(debug=False, host="0.0.0.0", port="8080")
