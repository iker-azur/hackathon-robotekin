## 🤖 SO-101 Autonomous Agent

Proyecto experimental de robótica desarrollado con LeRobot y el brazo robótico SO-101.

El objetivo es preparar un proyecto para el Hackathon de Robotekin 2026, desarrollándolo previamente en simulación y desplegándolo posteriormente sobre robots reales durante el hackathon.

La idea general es construir un sistema robótico capaz de:

VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE

El proyecto contempla dos casos de uso.

#### 🦾 Caso de uso 1 — Autonomous Agent

Un único SO-101 recibe una tarea y trata de completarla de forma autónoma.

Por ejemplo:

"Prepara una caja con una pieza roja y dos azules."

El sistema deberá:

Observar los objetos disponibles.

Entender la petición.

Crear un plan.

Ejecutar las acciones necesarias.

Comprobar el resultado.

Detectar posibles errores.

Replanificar si algo cambia.

Ejemplo

Usuario
   │
   │ "Mete una pieza roja y dos azules en la caja"
   ▼
Percepción
   │
   ▼
Planificación
   │
   ▼
SO-101
   │
   ├── Coger pieza roja
   ├── Coger pieza azul
   └── Coger pieza azul
   │
   ▼
Verificación
   │
   ▼
Pedido completado

El sistema también deberá poder reaccionar ante cambios inesperados.

Por ejemplo, si alguien mueve una pieza mientras el robot está trabajando:

Intentar coger pieza
        ↓
La pieza ha cambiado de posición
        ↓
La acción falla
        ↓
Observar de nuevo
        ↓
Encontrar la nueva posición
        ↓
Actualizar el plan
        ↓
Continuar

Primer objetivo

El primer objetivo será mucho más sencillo:

Conseguir que un SO-101 simulado coja un cubo y lo coloque en una zona determinada.

A partir de ahí se irá aumentando progresivamente la autonomía del sistema.

#### 🤖 Caso de uso 2 — Multi-Robot

Como extensión, el sistema podrá trabajar con varios SO-101 simultáneamente.

Un planificador central conocerá los robots disponibles y distribuirá las tareas entre ellos.

                  Tarea
                    │
                    ▼
               Planificador
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      SO-101 #1  SO-101 #2  SO-101 #3
          │         │         │
          └─────────┼─────────┘
                    ▼
               Verificación
                    │
                    ▼
               Replanificación

El objetivo es que el sistema pueda adaptarse automáticamente al número de robots disponibles.

1 robot
   ↓
Ejecución secuencial

3 robots
   ↓
Ejecución en paralelo

Un robot falla
   ↓
Redistribuir sus tareas
   ↓
Continuar

Este caso de uso se desarrollará principalmente en simulación y se probará con varios robots físicos únicamente si están disponibles durante el hackathon.

#### 🧪 Estrategia

Todo el desarrollo inicial se realizará mediante simulación.

SO-101 simulado
       ↓
Teleoperación
       ↓
Recoger demostraciones
       ↓
Entrenar políticas
       ↓
Ejecutar tareas autónomamente
       ↓
SO-101 real

La arquitectura intentará separar la lógica del sistema del robot utilizado.

De esta forma podremos trabajar inicialmente con:

SimulatedSO101

y posteriormente sustituirlo por:

RealSO101

sin modificar el resto del sistema.
