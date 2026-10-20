# RailGuard

Мы представляем два интерфейса для прототипа нашего решения: веб-приложение для работы с записями LiDAR в браузере и ROS 2-ноду для обработки облака точек и просмотра результатов в RViz. Оба используют общий код модели и фильтрации в `railpath/`.

## Структура репозитория

- `obstacle_app/` — веб-приложение.
- `railguard/` — ROS 2-нода, её конфигурация и настройки RViz.
- `railpath/` — общая модель и фильтрация препятствий.
- `data/`, `weights/`, `outputs/` — записи, веса модели и результаты.
- `compose.yaml` — запуск обоих интерфейсов.

Команды выполняйте из корня репозитория. Нужны Docker, NVIDIA GPU и NVIDIA Container Toolkit.

### Веса

Веса распространяются отдельно (каталоги в `.gitignore`). Два комплекта: 200 м (модель оси + рефайнер 200 м) и 400 м
(та же модель оси + рефайнер 400 м). Раскладка:

| Каталог | Кто читает | Содержимое |
| --- | --- | --- |
| `weights/` | ROS-нода (копируется в образ и монтируется поверх) | `path/best.pt`, `refine/last.pt`, `refine/calib.json` — комплект, с которым запускается нода; `RAILPATH_ROWS` в `compose.yaml` должен совпадать с рефайнером (200 или 400) |
| `200m_weights/` | веб-приложение, профиль Pandar128 (`/lidar_points`) | та же структура |
| `400m_weights/` | веб-приложение, профиль Falcon (`/iv_points`) | та же структура |

Распакованные ROS bag с `metadata.yaml` кладите в `data/<имя>/`.

### Настройка хоста (один раз)

Нода и bag обмениваются сканами по UDP (~7 МБ на скан, 10 Гц). Со стандартными буферами ядра (208 КБ) большая часть
сканов теряется (`RcvbufErrors` в `/proc/net/snmp` растут), поэтому поднимите лимиты:

```sh
sudo sysctl -w net.core.rmem_max=268435456 net.core.rmem_default=268435456 net.core.wmem_max=268435456 net.core.wmem_default=268435456
```

(чтобы сохранить после перезагрузки — те же строки в `/etc/sysctl.d/99-railguard.conf`).

## Веб-приложение

```sh
mkdir -p data outputs
docker compose up -d --build web
```

Откройте [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Начальная скорость KISS-ICP

При детекции в веб-приложении используются начальные скорости из настроек генерации GT. Это начальная оценка движения вперёд (вдоль −Y), а не постоянная скорость: дальнейшее движение оценивает KISS-ICP. Интервал в таблице используется для начального смещения: `скорость × интервал`.

| Имя каталога bag             | Начальная скорость, м/с | Интервал инициализации, с |
| --------------------------------------- | -----------------------------------------: | ----------------------------------------------: |
| `roundT_doubleT`                      |                                         10 |                                          0.0805 |
| `roundT_pressureGate_roundT`          |                                         10 |                                          0.1102 |
| `roundT_squareT_pressureGate_squareT` |                                         10 |                                          0.1026 |
| `squareT_platform_squareT_switch`     |                                         15 |                                          0.1043 |
| `doubleT_platform`                    |                                          0 |                                          0.1149 |
| `new_data`                            |                                          0 |                                          0.1013 |
| `doubleT_obstacle`                    |                                          0 |                                          0.1001 |
| `cloud_with_fake_obj`                 |                                          0 |                                             0.1 |
| `synthetic_switch_cloud_only`         |                                          0 |                                             0.1 |
| `synthetic_tunnels_cloud_only`        |                                         10 |                                             0.1 |
| `synthetic_platform_cloud_only`       |                                         16 |                                             0.1 |

Пресеты заданы в [obstacle_app/gt_odometry.py](obstacle_app/gt_odometry.py) и выбираются по точному имени каталога записи. Для известных записей скорость подставляется автоматически и недоступна для редактирования. Для экспортированных bag исходное имя берётся из поля `source_bag` файла `obstacle_scenario.json`.

Для неизвестной записи поле **Detection initial speed · m/s** остаётся пустым: перед включением детекции нужно явно указать скорость. Если поезд в начале записи стоит, введите `0`. Переименование исходного каталога отключает автоматический выбор его пресета. Для записей без пресета интервал инициализации равен 0.1 с.

Эти же пресеты использует ROS-нода (`RAILGUARD_BAG`, см. ниже); параметры симуляции настраиваются отдельно.

## ROS-нода и RViz

Нода запускается в одном из двух режимов. Переменные окружения `docker compose` попадают в аргументы launch-файла:

| Переменная | Значение | Режим |
| --- | --- | --- |
| `RAILGUARD_ODOMETRY` | `true` (по умолчанию) — с одометрией; `false` — без | оба |
| `RAILGUARD_BAG` | имя каталога записи → пресет начальной скорости KISS-ICP из таблицы выше | с одометрией |
| `RAILGUARD_INITIAL_SPEED` | начальная скорость, м/с (для записи без пресета или при старте с середины записи) | с одометрией |
| `RAILGUARD_INPUT_TOPIC` | топик облака точек (по умолчанию из `railguard/config/railguard.yaml`) | оба |

### Режим 1: с одометрией (по умолчанию)

Из одного launch-файла запускаются `railguard_odometry` (потоковый KISS-ICP, публикует `/railguard/odometry`) и
`railguard_node` (модель оси, кандидаты по одному скану, подтверждение препятствия по нескольким сканам с помощью
одометрии — та же процедура подтверждения и та же таблица начальных скоростей, что в веб-приложении). Регистрация
KISS-ICP в ноде идёт в онлайн-режиме (каждая 8-я точка, без deskew): офлайн-настройки приложения не успевают за 10 Гц.

Обязательно задайте начальную скорость: пресетом (`RAILGUARD_BAG`, для записи, воспроизводимой с начала) или числом
(`RAILGUARD_INITIAL_SPEED`; `0`, если поезд стоит; при старте с середины записи — фактическая скорость в этой точке).
Без того или другого launch завершается с ошибкой.

```sh
xhost +local:root
RAILGUARD_BAG=new_data docker compose up -d --build node          # пресет: скорость 0 м/с
RAILGUARD_INITIAL_SPEED=10 docker compose up -d node               # запись без пресета или старт с середины
docker compose logs -f node
```

### Режим 2: без одометрии

Только нода детекции: препятствия по одному скану по «немедленному» правилу (группа > 50 уникальных точек), без
подтверждения по нескольким сканам. Пресет и скорость не нужны.

```sh
RAILGUARD_ODOMETRY=false docker compose up -d node
```

### RViz и воспроизведение

Те же параметры можно задать вручную: `ros2 launch railguard railguard.launch.py bag:=new_data odometry:=true input_topic:=/lidar_points`.
В отдельных терминалах откройте RViz и воспроизведите bag:

```sh
docker compose exec node bash -ic 'rviz2 -d /opt/railguard/ws/install/railguard/share/railguard/rviz/railguard.rviz'
docker compose exec node bash -ic 'ros2 bag play /bags/new_data_val'
```

Смена режима или записи — перезапуск с другими переменными: `RAILGUARD_BAG=synthetic_platform_cloud_only docker compose up -d node`.

### Выходные топики ноды

| Топик                       | Тип                            | Содержимое                                                                                                                       |
| -------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `/railguard/centerline`        | `nav_msgs/msg/Path`             | Предсказанная ось пути.                                                                                                |
| `/railguard/rails`             | `sensor_msgs/msg/PointCloud2`   | Точки рельсов.                                                                                                                 |
| `/railguard/obstacle_candidates` | `sensor_msgs/msg/PointCloud2` | Кандидаты по одному скану (до подтверждения одометрией).                                                  |
| `/railguard/odometry`          | `nav_msgs/msg/Odometry`         | Позы KISS-ICP (`frame_id` меняется при каждом сбросе трека).                                              |
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

## Загрузка образов из архива

```sh
docker load -i railguard-images.tar.zst
```
