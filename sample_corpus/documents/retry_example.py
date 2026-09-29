def fetch_with_retries(client, url, max_attempts=3):
    """Request a URL repeatedly until it succeeds or the attempt limit is reached."""
    for attempt in range(max_attempts):
        try:
            return client.get(url)
        except TimeoutError:
            if attempt == max_attempts - 1:
                raise
