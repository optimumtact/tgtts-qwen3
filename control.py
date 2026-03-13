import argparse
import json
import os
import random
import sys
import time

import requests


def get_auth_header():
    token = os.getenv("TTS_AUTHORIZATION_TOKEN", "vote_goof_2024")
    return {"Authorization": token}


def generate_random_message(word_count=6):
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", 
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", 
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", 
        "xray", "yankee", "zulu", "the", "quick", "brown", "fox", "jumps", 
        "over", "lazy", "dog", "mission", "accomplished", "status", "report", "incoming",
    ]
    return " ".join(random.choice(words) for _ in range(word_count))


def toggle_logging(args):
    url = f"http://{args.host}:{args.port}/toggle-logging"
    try:
        print(f"Requesting logging toggle at {url}...")
        response = requests.get(url, headers=get_auth_header())
        if response.status_code == 200:
            print("Successfully toggled logging:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error toggling logging: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to API: {e}")


def generate_audio(args):
    base_url = f"http://{args.host}:{args.port}"

    # 1. Fetch voices
    try:
        print("Fetching available voices...")
        v_resp = requests.get(f"{base_url}/tts-voices", headers=get_auth_header())
        if v_resp.status_code != 200:
            print(f"Failed to get voices: {v_resp.status_code}")
            return
        voices = v_resp.json()
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return

    # 2. Ask user for voice
    print("\nAvailable voices:")
    for i, v in enumerate(voices):
        print(f"{i+1}. {v}")

    try:
        choice_input = input(f"\nSelect a voice (1-{len(voices)}): ")
        choice = int(choice_input) - 1
        voice = voices[choice]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # 3. Ask for text
    text = input("Enter text to generate: ")
    if not text:
        print("Text cannot be empty.")
        return

    # 4. Preparation
    identifier = f"cli_{os.getpid()}"
    filters = args.filters
    if args.radio:
        if "radio" not in filters:
            filters = f"{filters}|radio" if filters else "radio"

    params = {
        "voice": voice,
        "pitch": args.pitch,
        "special_filters": filters,
        "identifier": identifier,
    }

    # 5. Call TTS endpoint
    print(f"Generating base audio for '{voice}' (ID: {identifier})...")
    try:
        response = requests.get(
            f"{base_url}/tts",
            headers=get_auth_header(),
            params=params,
            json={"text": text},
            stream=True,
        )

        if response.status_code == 200:
            filename = f"output_{identifier}.ogg"
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Success! Base audio saved to {filename}")
        else:
            print(f"Error generating base audio: {response.status_code}")
            print(response.text)
            return

        # 6. If radio requested, fetch from radio endpoint
        if args.radio:
            print(f"Fetching radio processed version for ID: {identifier}...")
            radio_params = {"identifier": identifier}
            r_response = requests.get(
                f"{base_url}/tts-radio",
                headers=get_auth_header(),
                params=radio_params,
                stream=True,
            )

            if r_response.status_code == 200:
                radio_filename = f"output_radio_{identifier}.ogg"
                with open(radio_filename, "wb") as f:
                    for chunk in r_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Success! Radio version saved to {radio_filename}")
            else:
                print(f"Error fetching radio version: {r_response.status_code}")
                print(r_response.text)

    except Exception as e:
        print(f"Request failed: {e}")


def generate_load(args):
    base_url = f"http://{args.host}:{args.port}"

    # 1. Fetch voices
    try:
        print("Fetching voices for load test...")
        v_resp = requests.get(f"{base_url}/tts-voices", headers=get_auth_header())
        v_resp.raise_for_status()
        voices = v_resp.json()
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return

    print(f"Starting load test on {base_url}...")
    loop_count = 0

    try:
        while True:
            if args.max and loop_count >= args.max:
                print(f"\nReached max loops ({args.max}). Stopping.")
                break

            loop_count += 1
            voice = random.choice(voices)
            text = generate_random_message()
            mode = random.choice(["tts", "blips"])
            use_radio = random.random() > 0.5
            identifier = f"load_{os.getpid()}_{loop_count}"

            print(f"[{loop_count}] Mode: {mode} | Radio: {use_radio} | Voice: {voice} | ID: {identifier}")

            try:
                # Step 1: Base request
                if mode == "tts":
                    endpoint = "/tts"
                    filters = "radio" if use_radio else ""
                    payload = {"text": text}
                    params = {
                        "voice": voice,
                        "identifier": identifier,
                        "special_filters": filters,
                    }
                else:
                    endpoint = "/tts-blips"
                    payload = {"text": text}
                    params = {"voice": voice, "identifier": identifier}

                resp = requests.get(
                    f"{base_url}{endpoint}",
                    headers=get_auth_header(),
                    params=params,
                    json=payload,
                )
                if resp.status_code >= 400:
                    print(f"  [ERROR] {endpoint} returned {resp.status_code}: {resp.text[:100]}")

                # Step 2: Optional Radio request
                if use_radio and resp.status_code == 200:
                    radio_endpoint = "/tts-radio" if mode == "tts" else "/tts-blips-radio"
                    r_resp = requests.get(
                        f"{base_url}{radio_endpoint}",
                        headers=get_auth_header(),
                        params={"identifier": identifier},
                    )
                    if r_resp.status_code >= 400:
                        print(f"  [ERROR] {radio_endpoint} returned {r_resp.status_code}: {r_resp.text[:100]}")

            except Exception as e:
                print(f"  [EXCEPTION] {e}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nLoad test stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="TTS API Control Script")
    parser.add_argument("--host", default="localhost", help="API Host (default: localhost)")
    parser.add_argument("--port", default="5002", help="API Port (default: 5002)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Toggle Logging Command
    subparsers.add_parser("toggle-logging", help="Toggle logging level (INFO/DEBUG)")

    # Generate Audio Command
    gen_parser = subparsers.add_parser("generate", help="Generate audio from text")
    gen_parser.add_argument("--pitch", default="0", help="Pitch shift steps (default: 0)")
    gen_parser.add_argument("--filters", default="", help="Special filters (e.g. 'silicon', 'radio')")
    gen_parser.add_argument("--radio", action="store_true", help="Use radio effect endpoint")

    # Generate Load Command
    load_parser = subparsers.add_parser("load", help="Generate synthetic load")
    load_parser.add_argument("--max", type=int, help="Maximum number of requests (loops)")

    args = parser.parse_args()

    if args.command == "toggle-logging":
        toggle_logging(args)
    elif args.command == "generate":
        generate_audio(args)
    elif args.command == "load":
        generate_load(args)


if __name__ == "__main__":
    main()
