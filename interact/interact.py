import requests
import json
import argparse


def interact_with_server(prompt, image_urls, server_url):
    """
    Send a POST request to the server with the given prompt and image URLs.
    """
    payload = {
        "prompt": prompt,
        "image_urls": image_urls
    }

    try:
        response = requests.post(f"{server_url}/submit", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the server: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Interact with the LLM server")
    parser.add_argument("--server", default="http://localhost:5000",
                        help="Server URL (default: http://localhost:5000)")
    args = parser.parse_args()

    print("Welcome to the LLM Interaction CLI!")
    print(f"Server URL: {args.server}")
    print("Type 'quit' or 'exit' to end the session.")

    while True:
        prompt = input("\nEnter your prompt: ").strip()

        if prompt.lower() in ['quit', 'exit']:
            break

        image_urls = []
        while True:
            url = input(
                "Enter an image URL (or press Enter if done): ").strip()
            if url:
                image_urls.append(url)
            else:
                break

        if not image_urls:
            print("Warning: No image URLs provided. The model may not work as expected.")

        print("\nSending request to the server...")
        response = interact_with_server(prompt, image_urls, args.server)

        if response:
            print("\nServer Response:")
            print(response)
        else:
            print("\nFailed to get a response from the server.")

    print("Thank you for using the LLM Interaction CLI. Goodbye!")


if __name__ == "__main__":
    main()
