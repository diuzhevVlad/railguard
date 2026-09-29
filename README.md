# RailGuard

Команды выполняйте из корня репозитория. Нужны Docker, NVIDIA GPU и NVIDIA Container Toolkit. Положите веса в `weights/path/best.pt`, `weights/refine/last.pt`, `weights/refine/calib.json`, а распакованные ROS bag с `metadata.yaml` — в `data/<имя>/`.

## Веб-приложение

```sh
mkdir -p data outputs
docker compose up -d --build web
```

Откройте [http://127.0.0.1:8000](http://127.0.0.1:8000).

## ROS-нода и RViz

Запустите ноду:

```sh
xhost +local:root
docker compose up -d --build node
docker compose logs -f node
```

В отдельных терминалах откройте RViz и воспроизведите bag:

```sh
docker compose exec node bash -ic 'rviz2 -d /opt/railguard/ws/install/railguard/share/railguard/rviz/railguard.rviz'
docker compose exec node bash -ic 'ros2 bag play /bags/doubleT_obstacle'
```

Топик `input_topic` в [railguard.yaml](railguard/config/railguard.yaml) должен совпадать с топиком облака точек в bag. Для `doubleT_obstacle` установите в RViz `Fixed Frame: lidar_livox`. После изменения YAML примените параметры без пересборки:

```sh
docker compose stop node
docker compose start node
```

Остановить сервисы: `docker compose stop web node`.
