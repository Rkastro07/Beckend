# Modelos do Plan-to-BIM gratuito

- `walls.pt`: segmentação de paredes treinada para a fusão com a geometria 2D.
- `openings.pt`: detecção de portas e janelas alinhada ao dataset de plantas.

O contêiner usa os dois modelos no modo `hybrid-2d-yolo-fusion`, sempre em CPU
no Cloud Run. Eles não dependem de OpenAI, DeepSeek ou outro serviço de LLM.

SHA-256:

- `walls.pt`: `8F90A361EB26A1FD344DBCB31FB70AA0AA83B4F4390DCD44460CC988838358ED`
- `openings.pt`: `EE81C32FCF631C260FB7D94C594BBE3B97F67C276897CDE33D157689FF54633A`
