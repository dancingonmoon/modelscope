import replicate
# https://github.com/replicate/replicate-python#readme
output = replicate.run(
    "cjwbw/video-retalking:db5a650c807b007dc5f9e5abe27c53e1b62880d1f94d218d27ce7fa802711d67",
    input={
        "face": "https://replicate.delivery/pbxt/Jnm95KgYvAQIHlR0tg8rbWHweReTtCYp42Drl7dMNtHXaTNR/3.mp4",
        "input_audio": "https://replicate.delivery/pbxt/JnkUjVcUPLreS4x7ZXXQuCY7qVcLLDNxOeRAsHRi7qj79xBk/1.wav"
    }
)
print(output)