This is a datset of single-chords in MAESTRO datset.

list[
	{
                    "name": str,
                    "audio": list[float],
                    "onset": int,
                    "offset": int,
                    "pitches": list[int],
                    "velocities": list[int]
	}
]

For piano: lowest pitch is 21, pitch range is 88, velocities: [0,127]
For MAESTRO dataset: onset/offset 960 units = 1 second
Sample rate: 44100Hz