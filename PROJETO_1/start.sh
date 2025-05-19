#!/bin/bash



# Iniciar o servidor com o Waitress (mais estável para produção no Render)
waitress-serve --host 0.0.0.0 --port $PORT main:app