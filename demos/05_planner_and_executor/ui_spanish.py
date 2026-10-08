"""Spanish display text; raw simulator diagnostics remain in the session log."""
ACTIONS = {"above":"aproximar", "down":"bajar", "lower":"bajar", "close":"cerrar pinza",
           "lift":"elevar", "carry":"trasladar", "release":"soltar", "retreat":"retirar", "open":"abrir pinza"}
INTENTS = {"pick":"coger", "pick_and_place":"coger y colocar", "sort":"clasificar",
           "clear":"recoger", "sequence":"secuencia", "give":"entregar", "stop":"parar",
           "cancel":"cancelar", "question":"pregunta", "greeting":"saludo", "off_topic":"otra conversación",
           "not_doable":"no se puede realizar", "repeat":"repetir", "answer_yes":"respuesta afirmativa",
           "answer_no":"respuesta negativa", "answer_value":"respuesta"}


def findings(rows):
    labels = {"placement":"Comprueba el espacio de colocación: la cavidad debe dejar margen para el error de cámara.",
              "release":"La altura de caída puede provocar rebotes; comprueba la colocación final.",
              "reach":"Hay una pieza fuera del alcance o del área de trabajo.",
              "camera":"Esta cámara puede quedar oculta por el brazo; se recomienda la cámara superior.",
              "grasp":"Comprueba que el tamaño de la pieza permite un agarre físico con la pinza.",
              "grasp envelope":"El servo comprueba el centrado de la pinza y rechaza posiciones inaccesibles."}
    translated=[]
    for level, text in rows:
        category=text.split(":",1)[0]
        message=labels.get(category, "Las propiedades físicas y la calibración de cámara deben comprobarse en el montaje real.")
        if (level,message) not in translated: translated.append((level,message))
    return translated
