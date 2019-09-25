import json

if __name__ == "__main__":
	with open("./example_sentences.json") as f:
		examples = json.load(f)
	with open("./sentences.txt", "w") as f:
		for word in examples.values():
			for sentence_data in word:
				print(sentence_data["string"], file=f)