import numpy as np
import cv2


class SnakeDataImporter:
    def __init__(self, play_video=False):
        self.file_content = self.read_txt_file("snake_gameplay.txt")
        self.snake_video, self.snake_input = self.convert_txt_file_to_video(self.file_content, 42, 28)
        if play_video:
            self.show_video(self.snake_video, self.snake_input)
        pass

    def read_txt_file(self, file_name):
        with open("../../gameplay-data/snake-gameplay/{}".format(file_name), "r") as txt_file:

            content = txt_file.read().split("\n")
            for i, line in enumerate(content):
                content[i] = content[i][:-1]
        return content

    def convert_txt_file_to_video(self, content, width, height):
        video_array = np.full((len(content), height + 4, width + 4), 0.0001)
        input_array = np.full((len(content), 4), 0)

        for i, frame in enumerate(content):
            content[i] = frame.split(":")

            for j, part in enumerate(content[i]):
                part = part.split(",")

                if j == 0:
                    if part[0] == "up":
                        input_array[i][0] = 1
                    if part[0] == "down":
                        input_array[i][1] = 1
                    if part[0] == "left":
                        input_array[i][2] = 1
                    if part[0] == "right":
                        input_array[i][3] = 1

                if j == 1:
                    video_array[i][height - int(part[1])][int(part[0])] = 1

                if j == 2:
                    video_array[i][height - int(part[1])][int(part[0])] = 0.75

                if j > 2:
                    video_array[i][height - int(part[1])][int(part[0])] = 0.5

        return video_array, input_array

    def show_video(self, video_data, input_data, result=None, playback_speed=10):
        print(video_data.shape)
        for i, frame in enumerate(video_data):
            # print("Recorded inputs: {}".format(input_data[i]))
            print("Recorded inputs: {}".format(np.round(input_data[i], 2)))
            if result is not None:
                # print("Model prediction: {}".format(result[i]))
                # print("Model prediction: {}".format(np.round(result[i], 2)))
                print("Model prediction: {}".format(np.round(result[i], 2)))

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1000, 1000)
            cv2.imshow("image", frame)

            if cv2.waitKey(playback_speed) & 0xFF == ord("q"):
                pass
        cv2.destroyAllWindows()

    def get_data(self):
        return self.snake_video, self.snake_input


# importer = SnakeDataImporter(True)
