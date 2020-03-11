import cv2
import csv


class DataLoader:
    def __init__(self, video_name, input_name, show_video=False, playback_speed=1, compression_percentage=100):
        self.video = self.load_video(video_name)
        self.input_list = self.load_input(input_name)

        self.video_compressed = self.compress_video(self.video, compression_percentage)

        # Debug: Show the video while it's loading to check for input accuracy
        if show_video:
            self.play_video(self.video_compressed, self.input_list, playback_speed)

        pass

    # Loads the video from the gameplay-data video file
    # @param name: The name of the video file
    # @return video_frame: A list containing all the frame data of the video
    def load_video(self, name):
        # Read the video file
        cap = cv2.VideoCapture("../../gameplay-data/video/{}".format(name))
        video_frames = []

        # Add every frame to the video_frames list
        while cap.isOpened():
            ret, frame = cap.read()
            video_frames.append(frame)

            if not ret:
                break

        cap.release()

        return video_frames

    # Loads the input data from the gameplay-data csv file
    # @param name: The name of the csv file
    # @return input_keys: A list containing all the input keys for each frame
    def load_input(self, name):
        input_keys = []

        # Read the csv file
        with open("../../gameplay-data/input/{}".format(name)) as csv_file:
            csv_reader = csv.reader(csv_file)

            # Only take every other row since the csv file in bloated with empty rows
            for i, row in enumerate(csv_reader):
                if i % 2 == 0:
                    input_keys.append(row)
        return input_keys

    def compress_video(self, video, compression_percentage):
        if compression_percentage == 100:
            return video

        compressed_video = []

        for frame in video:
            if frame is not None:
                width = int(frame.shape[1] * compression_percentage / 100)
                height = int(frame.shape[0] * compression_percentage / 100)
                dim = (width, height)

                resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                compressed_video.append(resized_frame)
        return compressed_video

    # Play the video and the corresponding input keys to check if they line up properly
    def play_video(self, video, input_keys, playback_speed):
        # Set the timeout between each frame
        speed = 1000 / playback_speed
        speed = int(speed)

        # Play the video
        for i, frame in enumerate(video):
            if i < len(input_keys) - 1:
                cv2.imshow("Video", frame)
                print(input_keys[i])

            if cv2.waitKey(speed) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    # Get the video and corresponding input data
    def get_data(self):
        video = self.get_video()
        input = self.get_input()
        if len(video) > len(input):
            video = video[: len(input)]
        if len(input) > len(video):
            input = input[: len(video)]
        return video, input

    # Get the video frames
    def get_video(self):
        return self.video_compressed

    # Get the input keys list
    def get_input(self):
        return self.input_list


# data_loader = DataLoader("output_7.avi", "inputs_7.csv", True, 15, 40)
