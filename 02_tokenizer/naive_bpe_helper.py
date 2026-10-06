def split_into_pretokens(args):

    text, special_tokens = args

    if special_tokens:

        special_pattern = regex.compile(
            "("
            + "|".join(
                regex.escape(token)
                for token in sorted(
                    special_tokens,
                    key=len,
                    reverse=True
                )
            )
            + ")"
        )

        pieces = special_pattern.split(text)
        
    else:

        pieces = [text]
    pretokens = []

    for piece in pieces:

        # Ignore special tokens themselves
        if not piece or piece in special_tokens:
            continue

        pretokens.extend(
            GPT2_PATTERN.findall(piece)
        )

    return pretokens


def find_chunk_boundaries(
    file,
    desired_num_chunks,
    split_special_token: bytes,
):

    assert isinstance(
        split_special_token,
        bytes
    ), "Must represent special token as a bytestring"

    # Get total file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial boundaries
    chunk_boundaries = [
        i * chunk_size
        for i in range(
            desired_num_chunks + 1
        )
    ]

    # Make sure final boundary is exactly EOF
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    # Move boundaries forward until special token
    # is found
    for bi in range(
        1,
        len(chunk_boundaries) - 1
    ):

        initial_position = chunk_boundaries[bi]

        file.seek(initial_position)

        while True:

            mini_chunk = file.read(
                mini_chunk_size
            )

            # EOF
            if mini_chunk == b"":

                chunk_boundaries[bi] = file_size

                break

            # Find <|endoftext|>
            found_at = mini_chunk.find(
                split_special_token
            )

            if found_at != -1:

                chunk_boundaries[bi] = (
                    initial_position
                    + found_at
                )

                break

            initial_position += mini_chunk_size

    # Remove duplicate boundaries
    return sorted(
        set(chunk_boundaries)
    )

def read_chunk(args):

    filename, start, end = args

    with open(filename, "rb") as f:

        # Important: move to the chunk's starting byte
        f.seek(start)

        chunk = f.read(
            end - start
        ).decode(
            "utf-8",
            errors="ignore"
        )

    return chunk



def pretokenize_and_count(
    filename,
    boundaries
):

    word_freq = Counter()

    total_chunks = len(boundaries) - 1

    print(f"Starting pretokenization")
    print(f"Total chunks: {total_chunks}")

    for chunk_number, (start,end) in enumerate(zip(boundaries[:-1],boundaries[1:]),start=1):

        print(f"Processing chunk {chunk_number}/{total_chunks}")

        chunk = read_chunk((filename,start,end))

        print(f"Read {(end - start) / 1024**2:.1f} MB")

        tokens = split_into_pretokens(
            (chunk,
            ["<|endoftext|>"])
        )

        print(f"Pretokens in chunk: {len(tokens):,}")

        word_freq.update(tokens)

        print(f"  Unique pretokens so far: {len(word_freq):,}")

        del chunk
        del tokens

        print(f"  Chunk {chunk_number} complete.")
        print()

   
    print("Pretokenization COMPLETE")
    print(f"Unique pretokens: len(word_freq):,}")


    return word_freq


def get_pairs(vocab):

    results = Counter()

    for tokens, freq in vocab.items():

        for i in range(
            len(tokens) - 1
        ):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            results[pair] += freq

    return results


def merge_pairs(
    vocab,
    best_pair,
    new_id
):

    new_vocab = Counter()

    for tokens, freq in vocab.items():

        result = []

        i = 0

        while i < len(tokens):

            # Check whether current two tokens
            # are the pair we want to merge
            if (
                i < len(tokens) - 1
                and
                (
                    tokens[i],
                    tokens[i + 1]
                ) == best_pair
            ):

                result.append(new_id)

                i += 2

            else:

                result.append(
                    tokens[i]
                )

                i += 1

        new_vocab[
            tuple(result)
        ] += freq

    return new_vocab
