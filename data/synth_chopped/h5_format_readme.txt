This is a datset of single-chords in MAESTRO datset.

{
	"audio_x": np.array[int16],
	"pitches_x": np.array[int16],
	"velocities_x": np.array[int16],
	"onset_x": int16,
	"offset_x": int16
	...
}

For piano: lowest pitch is 21, pitch range is 88, velocities: [0,127]
For MAESTRO dataset: onset/offset 960 units = 1 second
Sample rate: 44100Hz

x = 1,2,...,2000