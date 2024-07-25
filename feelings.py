
import pygame
import random
import itertools
import numpy as np  # NumPyを使用して最適化を行う例
import tensorflow as tf

# 初期設定
window_size = 600
grid_size = 60  # グリッドサイズを適切に設定
cell_size = window_size // grid_size
bg_color = (255, 255, 255)
creature_color = (0, 255, 0)  # 生物の色
food_color = (255, 0, 0)
object_color = (0, 0, 255)
initial_energy = 1000  # 生物の初期エネルギーを調整
creature_size = 3  # 生物の大きさを3マスに設定
object_spawn_interval = 50  # ミリ秒
food_spawn_interval = 100  # ミリ秒
last_object_spawn_time = 0
last_food_spawn_time = 0
move_probability = 0.8  # 移動する確率

pygame.init()
window = pygame.display.set_mode((window_size, window_size))
clock = pygame.time.Clock()

# グリッドの初期化
grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]




# 各生物に予測器を追加する
class Creature:
    def __init__(self, x, y, energy=initial_energy, shape=None, move_probability=move_probability):
        self.x = x
        self.y = y
        self.energy = energy
        self.shape = shape if shape is not None else [(0, 0), (1, 0), (2, 0)]  # デフォルトは横一列
        self.head_color = (255, 0, 255)  # 頭の色
        self.move_probability = move_probability  # 移動確率

        self.layer_name = []
        self.weights = []
        self.biases = []
        self.neural_network = self.create_neural_network()  # 生物ごとに独自のニューラルネットワークを生成
        
    def create_neural_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='relu', input_shape=(1,), name='input_layer'),  # 入力層
            tf.keras.layers.Dense(3, activation='relu', name='hidden_layer1'),  # 隠れ層1
            tf.keras.layers.Dense(3, activation='relu', name='hidden_layer2'),  # 隠れ層2
            tf.keras.layers.Dense(1, name='output_layer')  # 出力層
        ])
    
        # モデルのコンパイル
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 重みとバイアスの初期値の設定
        for layer in model.layers:            
        # 初期値をセット
            self.layer_name = model.layers[0].name
            self.weights = model.layers[0].get_weights()[0]
            self.biases = model.layers[0].get_weights()[1]
            
        return model
    
    def set_move_probability(self, move_probability):
        self.move_probability = move_probability

    def set_weights(self, weights):
        self.weights = weights
    
    def set_biases(self, biases):
        self.biases = biases
    
    
    def get_neural_network_info(self):
        return self.layer_name, self.weights, self.biases
        
    def set_neural_network_info(self, layer):
        self.layer_name = layer.name  # 層名をセット
        self.weights, self.biases = layer.get_weights()  # 重みとバイアスをセット

    def move_object(self, object_x, object_y, dx, dy):
        # 物を動かす先の位置を計算
        new_x = object_x + dx
        new_y = object_y + dy
        # 新しい位置がグリッド内かつ空きマスであるかチェック
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size and grid[new_x][new_y] is None:
            grid[new_x][new_y] = 'object'  # 新しい位置に物を移動
            grid[object_x][object_y] = None  # 元の位置の物をクリア
            return True  # 物を動かすことに成功
        return False  # 物を動かすことができない

    def move(self):
        if random.random() < self.move_probability:  # 確率で移動
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dx, dy = random.choice(directions)

            # 生物の新しい位置を計算
            new_positions = [(self.x + dx + rel_x, self.y + dy + rel_y) for rel_x, rel_y in self.shape]
            can_move = True

            # 各パーツが移動可能かをチェック
            for new_x, new_y in new_positions:
                if not (0 <= new_x < grid_size and 0 <= new_y < grid_size):
                    can_move = False
                    break  # グリッド外には移動できない
                elif grid[new_x][new_y] == 'object':
                    # オブジェクトを動かせるか試みる
                    if not self.move_object(new_x, new_y, dx, dy):
                        can_move = False
                        break  # オブジェクトを動かせない場合、移動しない

            # 全てのパーツが移動可能なら、生物を移動させる
            if can_move:
                for rel_x, rel_y in self.shape:
                    current_x, current_y = self.x + rel_x, self.y + rel_y
                    grid[current_x][current_y] = None  # 現在位置をクリア

                self.x += dx
                self.y += dy

                # 新しい位置に生物を配置
                for rel_x, rel_y in self.shape:
                    new_x, new_y = self.x + rel_x, self.y + rel_y
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        grid[new_x][new_y] = self
                        if grid[new_x][new_y] == 'food':
                            self.energy += 10  # 食料を食べた場合、エネルギーを増やす


        
    def act(self):
        self.move()
        self.energy -= 1
        # エネルギーが尽きた場合、生物を削除
        if self.energy <= 0:
            self.die()

    def die(self):
        # 生物が占めていた全てのマスをクリア
        for rel_x, rel_y in self.shape:
            grid[self.x + rel_x][self.y + rel_y] = None

    def reproduce(self):
        if self.energy > initial_energy * 2:
            new_x, new_y = find_empty_space(self.size)
            if new_x is not None:
                offspring = Creature(new_x, new_y, initial_energy)
                grid[new_x][new_y] = offspring
                self.energy -= initial_energy


def find_empty_space(size):
    for _ in range(100):  # ランダムな位置を試す回数
        x, y = random.randint(0, grid_size - size), random.randint(0, grid_size - size)
        if all(grid[x + dx][y + dy] is None for dx, dy in itertools.product(range(size), repeat=2)):
            return x, y
    return None, None


def spawn_food():
    global last_food_spawn_time
    if pygame.time.get_ticks() - last_food_spawn_time >= food_spawn_interval:
        x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        if grid[x][y] is None:
            grid[x][y] = 'food'
            last_food_spawn_time = pygame.time.get_ticks()


def spawn_object():
    global last_object_spawn_time
    if pygame.time.get_ticks() - last_object_spawn_time >= object_spawn_interval:
        x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        if grid[x][y] is None:
            grid[x][y] = 'object'
            last_object_spawn_time = pygame.time.get_ticks()


def check_for_object_clusters():
    # 全てのセルをチェック
    for x in range(grid_size):
        for y in range(grid_size):
            if grid[x][y] == 'object':
                # 中心のマスから上下左右に隣接するマスをチェック
                adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                adjacent_objects = [pos for pos in adjacent_positions if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size and grid[pos[0]][pos[1]] == 'object']
                # 2つの隣接するマスが物質であれば、生物として認識
                if len(adjacent_objects) == 2:
                    convert_objects_to_creature(x, y, adjacent_objects)


def convert_objects_to_creature(x, y, adjacent_objects):
    new_creature = Creature(x, y, energy=initial_energy)
    # 中心のマスと2つの隣接するマスの位置を新しい生物の形状に変換
    new_creature.shape = [(0, 0)] + [(obj_x - x, obj_y - y) for obj_x, obj_y in adjacent_objects]
    # オブジェクトを消去し、新しい生物を配置
    grid[x][y] = new_creature
    for obj_x, obj_y in adjacent_objects:
        grid[obj_x][obj_y] = new_creature


def draw_grid():
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            if grid[x][y] == 'food':
                pygame.draw.rect(window, food_color, rect)
            elif grid[x][y] == 'object':
                pygame.draw.rect(window, object_color, rect)
            elif isinstance(grid[x][y], Creature):
                creature = grid[x][y]
                # 生物の体を描画
                pygame.draw.rect(window, creature_color, rect)
                # 頭の位置を描画（生物の左上のマスを頭とする）
                if x == creature.x and y == creature.y:
                    pygame.draw.rect(window, creature.head_color, rect)

def main():
    running = True
    while running:
       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        optimize_move_probabilities(grid)

        window.fill(bg_color)

        spawn_food()
        spawn_object()

        check_for_object_clusters()

        for x in range(grid_size):
            for y in range(grid_size):
                cell = grid[x][y]
                if isinstance(cell, Creature):
                    cell.act()
                    cell.reproduce()

        draw_grid()

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

##感情の基礎コード部分
def optimize_move_probabilities(grid):
    total_energy = sum(cell.energy for row in grid for cell in row if isinstance(cell, Creature))

    for row in grid:
        for cell in row:
            if isinstance(cell, Creature):
                current_move_probability = cell.move_probability

                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(3, activation='relu', input_shape=(1,), name='input_layer'),  # 入力層
                    tf.keras.layers.Dense(3, activation='relu', name='hidden_layer1'),  # 隠れ層1
                    tf.keras.layers.Dense(3, activation='relu', name='hidden_layer2'),  # 隠れ層2
                    tf.keras.layers.Dense(1, name='output_layer')  # 出力層
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')

                layer_name, weights, biases = cell.get_neural_network_info()
                model.get_layer('input_layer').set_weights([weights, biases])
                
                # model.fit()の前に、current_move_probabilityを適切な形状に変換する
                X = np.array([[current_move_probability]])  # 2次元の配列に変換
                
                # model.fit()に渡す前に、total_energyを適切な形状にブロードキャストする
                Y = np.array([[total_energy/100000]])  # 2次元の配列に変換
                                
                # モデルの学習（学習回数を減らす）
                model.fit(X, Y, epochs=1, verbose=0)  # verbose=0 で学習の進捗を非表示に
                
                    
                # モデルの学習後の重みとバイアスを生物に設定
                # 重みとバイアスを取得
                weights = model.get_layer(layer_name).get_weights()[0]
                biases = model.get_layer(layer_name).get_weights()[1]
                
                # 生物にセット
                cell.set_neural_network_info(model.get_layer(layer_name))
                
                # 生物が持つニューラルネットワーク情報を更新
                cell.set_weights(weights)
                cell.set_biases(biases)


                # モデルへの入力データを生成
                random_probabilities = np.random.rand(5, 1)  # ランダムな移動確率を生成
                predicted_energies = model.predict(random_probabilities) * 100000  # 予測値を計算    
                max_index = np.argmax(predicted_energies)  # 最大予測値のインデックスを取得
                max_predicted_energy = predicted_energies[max_index]  # 最大予測値を取得
                print("Max predicted energy:", max_predicted_energy)  # 最大予測値をプリント
                optimized_probability = random_probabilities[max_index][0]  # 最適化された確率を取得
                # 最適化された確率を設定
                cell.set_move_probability(optimized_probability)
                
    return grid


if __name__ == '__main__':
    main()
