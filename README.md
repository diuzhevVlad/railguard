# RailGuard

Мы представляем два интерфейса для прототипа нашего решения: веб-приложение для работы с записями LiDAR в браузере и ROS 2-ноду для обработки облака точек и просмотра результатов в RViz. Оба используют общий код модели и фильтрации в `railpath/`.

## Структура репозитория

- `obstacle_app/` — веб-приложение.
- `railguard/` — ROS 2-нода, её конфигурация и настройки RViz.
- `railpath/` — общая модель и фильтрация препятствий.
- `data/`, `weights/`, `outputs/` — записи, веса модели и результаты.
- `compose.yaml` — запуск обоих интерфейсов.

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
docker compose exec node bash -ic 'ros2 bag play /bags/doubleT_platform'
```

### Выходные топики ноды

| Топик                       | Тип                            | Содержимое                                                                                                                       |
| -------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `/railguard/centerline`        | `nav_msgs/msg/Path`             | Предсказанная ось пути.                                                                                                |
| `/railguard/rails`             | `sensor_msgs/msg/PointCloud2`   | Точки рельсов.                                                                                                                 |
| `/railguard/obstacles`         | `sensor_msgs/msg/PointCloud2`   | Точки обнаруженных препятствий.                                                                                |
| `/railguard/obstacle_detected` | `std_msgs/msg/Bool`             | `true`, если препятствие обнаружено.                                                                            |
| `/railguard/obstacle_distance` | `std_msgs/msg/Float32`          | Расстояние до ближайшего препятствия вдоль пути в метрах;`NaN`, если его нет. |
| `/railguard/nearest_obstacle`  | `visualization_msgs/msg/Marker` | Ближайшая точка — жёлтая сфера в RViz; удаляется, когда препятствий нет.            |

Топик `input_topic` в [railguard.yaml](railguard/config/railguard.yaml) должен совпадать с топиком облака точек в bag. Для `doubleT_obstacle` установите в RViz `Fixed Frame: lidar_livox` и соответствующий топик лидарных данных. После изменения YAML примените параметры без пересборки:

```sh
docker compose stop node
docker compose start node
```

Остановить сервисы: `docker compose stop web node`.
