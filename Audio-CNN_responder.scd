(
s.waitForBoot(
	{
		p = ProxySpace.new;
		p = ProxySpace.push;
		p.reshaping = \elastic;

			// audio samples here
		a = Buffer.read(s, '/path/to/audio/file.wav');
		b = Buffer.read(s, '/path/to/audio/file.wav');
		c = Buffer.read(s, '/path/to/audio/file.wav');
		d = Buffer.read(s, '/path/to/audio/file.wav');
		f = Buffer.read(s, '/path/to/audio/file.wav');

		~pitch = {
			|rate=1.01, bufnum|
			var trig = Coyote.kr(in:SoundIn.ar(0), fastMul:0.55, minDur:0.75, thresh:0.05);
			Pan2.ar(PlayBuf.ar(numChannels:2, bufnum:TChoose.kr(trig, [a, b, c, d]), rate:rate, trigger:trig, doneAction:0), level:0.7);
		};

		~timbre = {
			|rate=1.01|
			var fft = FFT(LocalBuf(2048), SoundIn.ar(0));
			var amp = Amplitude.kr(SoundIn.ar(0));
			var dissonance = SensoryDissonance.kr(fft, clamp:100.0);
			var trig = Coyote.kr(in:SoundIn.ar(0), fastMul:0.65, minDur:0.65, thresh:0.05);
			// dissonance.poll(5);
			Pan2.ar(PlayBuf.ar(numChannels:2, bufnum:f, rate:rate, trigger:dissonance>0.1, doneAction:0), level:0.7);
		};

		~pitch.fadeTime=0.5;
		~timbre.fadeTime=0.5;

		~input = {SoundIn.ar(0)};
		~input_w_delay = {HilbertFIR.ar(~input, LocalBuf(2048))};
		~input_w_delay.play;

		thisProcess.openUDPPort(1337);
		OSCdef.trace(false);
		OSCdef(\CNN_responder, {
			|msg|

			case

			//{DetectSilence.ar(SoundIn.ar([0,1]), 0.001, 0.75) == 1} {Ndef(\samplePlayerC).play; Ndef(\samplePlayerP).stop; Ndef(\samplePlayerT).stop;}

			{msg[1] == '[0 1]'} {~timbre.stop; ~pitch.play;}

			{msg[1] == '[1 0]'} {~pitch.stop; ~timbre.play;}

		},'/neuralnet1');

	}
)
)