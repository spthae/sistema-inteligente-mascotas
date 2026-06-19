import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Input, Flatten


carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_emociones = os.path.join(carpeta_actual, "emociones.csv")
ruta_salud = os.path.join(carpeta_actual, "salud.csv")
ruta_alertas = os.path.join(carpeta_actual, "alertas.csv")


data = pd.read_csv(ruta_emociones, sep=";", encoding="ISO-8859-1")
X_text = data["Comportamiento"].values
X_animal = data["Animal"].values
y_labels = data["Emocion"].values

encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y_labels)
y_onehot = to_categorical(y_encoded)

encoder_animal = LabelEncoder()
X_animal_encoded = encoder_animal.fit_transform(X_animal)

max_words = 100
max_len = 20
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_seq_padded = pad_sequences(X_seq, maxlen=max_len, padding="post")


input_text = Input(shape=(max_len,))
x = Embedding(input_dim=max_words, output_dim=16, input_length=max_len)(input_text)
x = LSTM(32)(x)

input_animal = Input(shape=(1,))
y = Embedding(input_dim=len(encoder_animal.classes_), output_dim=8, input_length=1)(
    input_animal
)
y = Flatten()(y)

combined = Concatenate()([x, y])
output = Dense(y_onehot.shape[1], activation="softmax")(combined)

model = Model(inputs=[input_text, input_animal], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    [X_seq_padded, X_animal_encoded],
    y_onehot,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
)


SINONIMOS_ANIMAL = {
    "canino": "perro",
    "perro": "perro",
    "felino": "gato",
    "gato": "gato",
    "conejo": "conejo",
    "pajaro": "pajaro",
    "ave": "pajaro",
}


def predecir_emocion(texto, animal):
    animal_input = animal.strip().lower()
    animal_mapeado = SINONIMOS_ANIMAL.get(animal_input, None)

    if animal_mapeado is None or animal_mapeado not in encoder_animal.classes_:
        return "Desconocida"

    animal_encoded = encoder_animal.transform([animal_mapeado])

    seq = tokenizer.texts_to_sequences([texto.lower()])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict([padded, animal_encoded], verbose=0)
    return encoder_y.inverse_transform([pred.argmax()])[0]


alertas_df = pd.read_csv(ruta_alertas, sep=";", encoding="latin1")
alertas_df.columns = alertas_df.columns.str.strip()  # Elimina espacios extras


def alerta_salud(comentario):
    comentarios_lower = comentario.lower()
    mensajes = []
    for _, row in alertas_df.iterrows():
        if row["Sintoma"].lower() in comentarios_lower:
            mensajes.append(row["Alerta"])
    return mensajes


def detector_salud():
    print("\n=== Detector de salud ===")
    salud_data = pd.read_csv(ruta_salud, sep=";", encoding="latin1")

    while True:
        busqueda = input("Busca por ID o Nombre de la mascota : ")
        if busqueda.lower() == "volver":
            break

        if busqueda.isdigit():
            mascota = salud_data[salud_data["ID Mascota"] == int(busqueda)]
        else:
            mascota = salud_data[salud_data["Nombre"].str.lower() == busqueda.lower()]

        if mascota.empty:
            print("Intenta de nuevo......\n")
            continue

        idx = mascota.index[0]
        print("\nInformación de la mascota:")
        print(
            mascota.loc[
                idx,
                [
                    "Nombre",
                    "Tipo de mascota",
                    "Raza",
                    "Sexo",
                    "Edad",
                    "Peso",
                    "Nivel de actividad",
                    "Comentarios del dueño",
                ],
            ]
        )

        editar = input("¿Deseas editar comentarios o nivel de actividad? (s/n): ")
        if editar.lower() == "s":
            nuevo_comentario = input("Nuevo comentario: ")
            nuevo_actividad = input("Nuevo nivel de actividad (Alta/Media/Baja): ")
            if nuevo_comentario.strip():
                salud_data.at[idx, "Comentarios del dueño"] = nuevo_comentario
            if nuevo_actividad.strip():
                salud_data.at[idx, "Nivel de actividad"] = nuevo_actividad
            salud_data.to_csv(ruta_salud, sep=";", index=False, encoding="latin1")
            print("Datos actualizados correctamente.\n")

        comentario_actual = salud_data.at[idx, "Comentarios del dueño"]
        mensajes_alerta = alerta_salud(comentario_actual)
        if mensajes_alerta:
            print("\n---------ALERTAS DE SALUD DETECTADAS!!!!:")
            for m in mensajes_alerta:
                print(f"- {m}")
            print(
                "\nCONSEJO: Si los síntomas persisten, consulta con tu veterinario.\n"
            )
        else:
            print("\nNo se detectaron síntomas preocupantes en los comentarios.\n")

        edad = int(mascota["Edad"].values[0])
        tipo = mascota["Tipo de mascota"].values[0].lower()
        actividad = mascota["Nivel de actividad"].values[0].lower()

        edad_humana = 0
        if tipo in ["perro", "canino"]:
            if edad <= 2:
                edad_humana = edad * 12.5
            else:
                edad_humana = 2 * 12.5 + (edad - 2) * 4
        elif tipo in ["gato", "felino"]:
            if edad <= 2:
                edad_humana = edad * 12.5
            else:
                edad_humana = 2 * 12.5 + (edad - 2) * 4
        elif tipo == "conejo":
            edad_humana = edad * 8
        vida_promedio = (
            15
            if tipo in ["perro", "canino"]
            else 16 if tipo in ["gato", "felino"] else 12
        )

        factor = 1.0
        if actividad in ["alta", "activa"]:
            factor = 1.1
        elif actividad in ["baja", "sedentaria"]:
            factor = 0.9

        vida_saludable_total = vida_promedio * factor
        vida_restante = max(0, vida_saludable_total - edad)

        print(
            f"{mascota['Nombre'].values[0]} tiene {edad} años ({edad_humana:.0f} años humanos aprox.)"
        )
        print(f"Tiempo de vida saludable restante: {vida_restante:.1f} años aprox.\n")


def agregar_mascota():
    print("\n=== Agregar nueva mascota ===")
    salud_data = pd.read_csv(ruta_salud, sep=";", encoding="latin1")

    nuevo_id = salud_data["ID Mascota"].max() + 1
    nombre = input("Nombre de la mascota: ")
    tipo = input("Tipo de mascota")
    sexo = input("Sexo")
    raza = input("Raza: ")
    edad = input("Edad: ")
    peso = input("Peso : ")
    actividad = input("Nivel de actividad : ")
    comentario = input("Comentario del dueño o veterinario: ")

    nueva_fila = {
        "ID Mascota": nuevo_id,
        "Nombre": nombre,
        "Tipo de mascota": tipo,
        "Sexo": sexo,
        "Raza": raza,
        "Edad": edad,
        "Peso": peso,
        "Nivel de actividad": actividad,
        "Comentarios del dueño": comentario,
    }

    salud_data = pd.concat([salud_data, pd.DataFrame([nueva_fila])], ignore_index=True)
    salud_data.to_csv(ruta_salud, sep=";", index=False, encoding="latin1")
    print(f"\nMascota '{nombre}' agregada correctamente con ID {nuevo_id}.\n")


# MENÚ PRINCIPAL
while True:
    print("\n=== MENU PRINCIPAL ===")
    print("1. Detector de emociones")
    print("2. Detector de salud")
    print("3. Agregar nueva mascota")
    print("4. Salir")

    opcion = input("Elige una opcion: ")

    if opcion == "1":
        print("\n=== Detector de emociones de mascotas ===")
        print("Escribe 'salir' para terminar.\n")
        while True:
            animal = input("Escribe el tipo de mascota: ")
            if animal.lower() == "salir":
                break
            comportamiento = input("Describe el comportamiento de tu mascota: ")
            if comportamiento.lower() == "salir":
                break
            emocion = predecir_emocion(comportamiento, animal)
            print(f"Emocion predicha: {emocion}\n")
    elif opcion == "2":
        detector_salud()
    elif opcion == "3":
        agregar_mascota()
    elif opcion == "4":
        print("\n¡Gracias por preocuparte por la emocion y salud de tu mascota!")
        break
    else:
        print("Intenta de nuevo......")
