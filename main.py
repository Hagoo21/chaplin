import torch
import hydra
import cv2
import time
from pipelines.pipeline import InferencePipeline
import numpy as np
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
import keyboard
from concurrent.futures import ThreadPoolExecutor
import os
import dotenv
from typing import List, Dict, Union

# Load environment variables from .env file
dotenv.load_dotenv()

# pydantic model for the chat output
class ChaplinOutput(BaseModel):
    list_of_changes: List[Dict[str, Union[str, int]]]
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 2
        self.fps = 25
        self.frame_interval = 1 / self.fps
        self.frame_compression = 85

        # initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please set it with your OpenAI API key.")
        self.openai_client = OpenAI(api_key=api_key)

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)
        print(f"📝 RAW VSR OUTPUT: '{output}'")

        # write the raw output
        print("⌨️  Writing raw output to keyboard...")
        keyboard.write(output)

        # shift left to select the entire output
        print("🔍 Selecting raw output for replacement...")
        
        cmd = ""
        for i in range(len(output)):
            cmd += 'shift+left, '
        cmd = cmd[:-2]
        keyboard.press_and_release(cmd)

        # perform inference on the raw output to get back a "correct" version
        print("🤖 Calling GPT-4o for text correction...")
        
        system_prompt = """

        <background>
        You are an assistant trained to post-process the output of a lipreading AI system. The system attempts to transcribe spoken words directly from video footage, but the resulting text often contains errors due to the limitations of visual-only input.
        </background>

        <objective>
        Your goal is to revise the transcript so that it accurately reflects the words likely spoken in the video, based only on the visual lip movement information.
        </objective>

        <inputformat>
        You will receive raw text in all capital letters, with no punctuation. This is the direct output of the lipreading model.
        </inputformat>

        <executionrules>
        You MUST follow these rules: 

        1. Correction Rules
        - If a word looks unusual, illogical, or out of context, assume it was mistranscribed and replace it with the most likely intended word based on the sentence structure and meaning.
        - You must not add entirely new words or elaborate on the sentence — only substitute obviously incorrect words.
        - Do not rewrite the sentence or change its structure — only individual word substitutions are allowed.
        - Maintain the original number of words whenever possible unless a word needs to be split (e.g., "alot" → "a lot") or merged (e.g., "every day" → "everyday") to make grammatical sense.

        2. Punctuation Rules
        - You must correctly punctuate all sentences.
        - Every sentence must end with a period (.), question mark (?), or exclamation mark (!).
        - Add commas, apostrophes, or quotation marks as needed for grammatical correctness.

        3. Casing
        - The final output must use standard English capitalization (i.e., not all-caps).
        - Proper nouns and sentence beginnings should be capitalized correctly.

        4. Output Format
        You must return a JSON object with the following two fields:
        4a. "corrected_text": the final, corrected, and punctuated version of the transcript.
        4b. list_of_changes": a list of word-level changes made. Each item must be an object with:
            -"original": the word as it appeared in the input.
            -"corrected": the word it was changed to.
            -"index": the 0-based index of the word in the input text.

        <exampleinput>
        I WENT TO THE STORE TO BYE SOME BREED AND CHESS
        </exampleoutput>

          "corrected_text": "I went to the store to buy some bread and cheese.",
        "list_of_changes": [
            {"original": "BYE", "corrected": "buy", "index": 6},
            {"original": "BREED", "corrected": "bread", "index": 7},
            {"original": "CHESS", "corrected": "cheese", "index": 9}
            ]
    <keybehaviors>
    -Work independently and systematically.
    -Be extremely precise in word substitution decisions.
    -Never ask for clarification or context — your decision must be based only on the given text.
    </keybehaviors>

        """
        user_message = f"Transcription:\n\n{output}"
        
        print(f"📤 Sending to GPT-4o: '{output}'")
        
        response = self.openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_message
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=400
        )
        print("✅ GPT-4o response received")

        # get only the corrected text
        print("🔧 Parsing GPT-4o response...")
        chat_output = ChaplinOutput.model_validate_json(
            response.choices[0].message.content)
        print(f"📝 Corrected text: '{chat_output.corrected_text}'")

        # if last character isn't a sentence ending (happens sometimes), add a period
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'
            print("🔧 Added missing sentence ending")

        # write the corrected text
        print("⌨️  Writing corrected text to keyboard...")
        keyboard.write(chat_output.corrected_text + " ")

        print("✅ Inference pipeline completed successfully")
        # return the corrected text and the video path
        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(0)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        print("🎥 Webcam feed started. Press 'alt' to start/stop recording, 'q' to quit.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting application...")
                # properly close any open video writer first
                if out is not None:
                    out.release()
                    out = None
                    print("📹 Closed video writer")
                
                # remove any remaining videos that were saved to disk
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        try:
                            os.remove(file)
                            print(f"🗑️  Cleaned up temporary file: {file}")
                        except PermissionError:
                            print(f"⚠️  Could not delete {file} - file may still be in use")
                        except Exception as e:
                            print(f"⚠️  Error deleting {file}: {e}")
                break

            current_time = time.time()

            # conditional ensures that the video is recorded at the correct frame rate
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    # frame compression
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(
                        buffer, cv2.IMREAD_GRAYSCALE)

                    if self.recording:
                        if out is None:
                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )
                            print(f"🎬 Started recording to: {output_path}")

                        out.write(compressed_frame)

                        last_frame_time = current_time

                        # circle to indicate recording, only appears in the window and is not present in video saved to disk
                        cv2.circle(compressed_frame, (frame_width -
                                                      20, 20), 10, (0, 0, 0), -1)

                        frame_count += 1
                    # check if not recording AND video is at least 2 seconds long
                    elif not self.recording and frame_count > 0:
                        print(f"🔍 Recording stopped. Frame count: {frame_count}, FPS: {self.fps}, Duration: {frame_count/self.fps:.1f}s")
                        if out is not None:
                            out.release()
                            out = None
                            print(f"⏹️  Stopped recording. Total frames: {frame_count}")
                            # Small delay to ensure video writer is fully closed
                            time.sleep(0.1)

                        # only run inference if the video is at least 2 seconds long
                        min_frames = self.fps * 2
                        if frame_count >= min_frames:
                            print(f"🚀 Submitting video for processing (duration: {frame_count/self.fps:.1f}s, min required: 2.0s)")
                            futures.append(self.executor.submit(
                                self.perform_inference, output_path))
                        else:
                            print(f"⚠️  Video too short ({frame_count/self.fps:.1f}s), skipping processing (minimum: 2.0s)")
                            try:
                                os.remove(output_path)
                                print(f"🗑️  Deleted short video: {output_path}")
                            except Exception as e:
                                print(f"⚠️  Error deleting short video {output_path}: {e}")

                        output_path = self.output_prefix + \
                            str(time.time_ns() // 1_000_000) + '.mp4'
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height),
                            False  # isColor
                        )

                        frame_count = 0

                    # display the frame in the window
                    cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

            # ensures that videos are handled in the order they were recorded
            for fut in futures:
                if fut.done():
                    result = fut.result()
                    # once done processing, delete the video with the video path
                    try:
                        os.remove(result["video_path"])
                        print(f"🗑️  Cleaned up processed video: {result['video_path']}")
                    except PermissionError:
                        print(f"⚠️  Could not delete processed video {result['video_path']} - file may still be in use")
                    except Exception as e:
                        print(f"⚠️  Error deleting processed video {result['video_path']}: {e}")
                    futures.remove(fut)
                else:
                    break

        # release everything
        print("🧹 Cleaning up resources...")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")

    def on_action(self, event):
        # toggles recording when alt key is pressed
        if event.event_type == keyboard.KEY_DOWN and event.name == 'alt':
            self.recording = not self.recording
            if self.recording:
                print("🔴 Recording STARTED - start speaking!")
            else:
                print("⏸️  Recording STOPPED - processing will begin...")


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    print("🚀 Starting Chaplin application...")
    chaplin = Chaplin()

    # hook to toggle recording
    print("⌨️  Setting up keyboard hooks...")
    keyboard.hook(lambda e: chaplin.on_action(e))

    # load the model
    print("🤖 Loading VSR model...")
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available(
        ) and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=True)
    print("✅ Model loaded successfully!")

    # start the webcam video capture
    print("🎬 Starting webcam capture...")
    chaplin.start_webcam()


if __name__ == '__main__':
    main()