"""Spanish display text; raw simulator diagnostics remain in the session log."""
ACTIONS = {"above":"aproximar", "down":"bajar", "lower":"bajar", "close":"cerrar pinza",
           "lift":"elevar", "carry":"trasladar", "release":"soltar", "retreat":"retirar", "open":"abrir pinza"}
INTENTS = {"pick":"coger", "pick_and_place":"coger y colocar", "sort":"clasificar",
           "clear":"recoger", "sequence":"secuencia", "give":"entregar", "stop":"parar",
           "cancel":"cancelar", "question":"pregunta", "greeting":"saludo", "off_topic":"otra conversación",
           "not_doable":"no se puede realizar", "repeat":"repetir", "answer_yes":"respuesta afirmativa",
           "answer_no":"respuesta negativa", "answer_value":"respuesta"}
# The same intents, as the verb of a sentence that says what the robot is about to do.
# A second map rather than a reuse of INTENTS because the two read differently: the log note is a
# label for a stage ("clasificar [cubo]"), the sentence is something a person reads at a glance
# ("separar un cubo rojo por color"). They cannot be the same words.
SAY = {"pick_and_place": "coger y colocar", "pick": "coger", "sort": "separar",
       "clear": "recoger", "sequence": "encadenar", "give": "entregar", "stop": "parar",
       "cancel": "cancelar", "question": "preguntar", "greeting": "saludar",
       "off_topic": "hablar de otra cosa", "not_doable": "no poder realizar",
       "repeat": "repetir", "answer_yes": "responder que sí",
       "answer_no": "responder que no", "answer_value": "responder con un valor"}
# What a sort ordered by, in the sentence.
ORDER = {"color": "por color", "kind": "por tipo"}


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
