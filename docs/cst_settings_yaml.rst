CST Settings Files
------------------

The text files saved out of CST beam simulations do not have much of the
critical metadata needed for UVBeam objects. This required metadata can be set
via keywords when the files are read in, but it is better for the metadata to be
specified once and carried with the data files. To that end, we developed a yaml
settings file specification to carry all the metadata. This format is very human
readable and writeable and we encourage using such a file as the best way to
ensure the metadata is preserved.

Required Fields
***************

The following are the required fields in a CST yaml settings file. The lists of
frequencies specifies the frequency in each filename, so the lists must be in
the same order (as must the feed_pol if it is a list):

- telescope_name (str)
- feed_name (str)
- feed_version (str)
- model_name (str)
- model_version (str)
- history (str)
- frequencies (list(float))
- cst text filenames (list(str)) -- path relative to yaml file location
- feed_pol (str) or (list(str))

Optional Fields
***************

The following are optional fields:

- ref_imp (float): beam model reference impedance
- sim_beam_type (str): e.g. 'E-farfield'
- any other field that contains useful information that should be propagated
  with the beam. These will go into the extra_keywords attribute (note that if the
  field names are more than 8 characters they will be truncated to 8 if the beam
  is written to a beamfits file).

Example Settings File
*********************

An example settings yaml file for a HERA Vivaldi feed simulation is shown below.
In this example, 'software', 'layout', and 'port_num' are extra fields that will
be propagated with the beam. Note that quotes are used around version numbers
that could be interpreted as floats and around filenames with spaces in them,
but other strings do not require quotes. The history field shows a way to make
a multi-line string (although this history is quite short, there could be many
lines of history information there.):

.. code-block:: yaml
    :caption: Example settings yaml file.

    telescope_name: HERA
    feed_name: Vivaldi
    feed_version: '1.0'
    model_name: Mecha design - dish - cables - soil
    model_version: '1.0'
    software: CST 2016
    history: |
        beams simulated in Nov 2018 by NF
    frequencies: [50e6, 51e6, 52e6, 53e6, 54e6, 55e6, 56e6, 57e6, 58e6, 59e6, 60e6, 61e6, 62e6, 63e6, 64e6, 65e6, 66e6, 67e6, 68e6, 69e6, 70e6, 71e6, 72e6, 73e6, 74e6, 75e6, 76e6, 77e6, 78e6, 79e6, 80e6, 81e6, 82e6, 83e6, 84e6, 85e6, 86e6, 87e6, 88e6, 89e6, 90e6, 91e6, 92e6, 93e6, 94e6, 95e6, 96e6, 97e6, 98e6, 99e6, 100e6, 101e6, 102e6, 103e6, 104e6, 105e6, 106e6, 107e6, 108e6, 109e6, 110e6, 111e6, 112e6, 113e6, 114e6, 115e6, 116e6, 117e6, 118e6, 119e6, 120e6, 121e6, 122e6, 123e6, 124e6, 125e6, 126e6, 127e6, 128e6, 129e6, 130e6, 131e6, 132e6, 133e6, 134e6, 135e6, 136e6, 137e6, 138e6, 139e6, 140e6, 141e6, 142e6, 143e6, 144e6, 145e6, 146e6, 147e6, 148e6, 149e6, 150e6, 151e6, 152e6, 153e6, 154e6, 155e6, 156e6, 157e6, 158e6, 159e6, 160e6, 161e6, 162e6, 163e6, 164e6, 165e6, 166e6, 167e6, 168e6, 169e6, 170e6, 171e6, 172e6, 173e6, 174e6, 175e6, 176e6, 177e6, 178e6, 179e6, 180e6, 181e6, 182e6, 183e6, 184e6, 185e6, 186e6, 187e6, 188e6, 189e6, 190e6, 191e6, 192e6, 193e6, 194e6, 195e6, 196e6, 197e6, 198e6, 199e6, 200e6, 201e6, 202e6, 203e6, 204e6, 205e6, 206e6, 207e6, 208e6, 209e6, 210e6, 211e6, 212e6, 213e6, 214e6, 215e6, 216e6, 217e6, 218e6, 219e6, 220e6, 221e6, 222e6, 223e6, 224e6, 225e6, 226e6, 227e6, 228e6, 229e6, 230e6, 231e6, 232e6, 233e6, 234e6, 235e6, 236e6, 237e6, 238e6, 239e6, 240e6, 241e6, 242e6, 243e6, 244e6, 245e6, 246e6, 247e6, 248e6, 249e6, 250e6]
    filenames: ['farfield_50MHz.txt', 'farfield_51MHz.txt', 'farfield_52MHz.txt', 'farfield_53MHz.txt', 'farfield_54MHz.txt', 'farfield_55MHz.txt', 'farfield_56MHz.txt', 'farfield_57MHz.txt', 'farfield_58MHz.txt', 'farfield_59MHz.txt', 'farfield_60MHz.txt', 'farfield_61MHz.txt', 'farfield_62MHz.txt', 'farfield_63MHz.txt', 'farfield_64MHz.txt', 'farfield_65MHz.txt', 'farfield_66MHz.txt', 'farfield_67MHz.txt', 'farfield_68MHz.txt', 'farfield_69MHz.txt', 'farfield_70MHz.txt', 'farfield_71MHz.txt', 'farfield_72MHz.txt', 'farfield_73MHz.txt', 'farfield_74MHz.txt', 'farfield_75MHz.txt', 'farfield_76MHz.txt', 'farfield_77MHz.txt', 'farfield_78MHz.txt', 'farfield_79MHz.txt', 'farfield_80MHz.txt', 'farfield_81MHz.txt', 'farfield_82MHz.txt', 'farfield_83MHz.txt', 'farfield_84MHz.txt', 'farfield_85MHz.txt', 'farfield_86MHz.txt', 'farfield_87MHz.txt', 'farfield_88MHz.txt', 'farfield_89MHz.txt', 'farfield_90MHz.txt', 'farfield_91MHz.txt', 'farfield_92MHz.txt', 'farfield_93MHz.txt', 'farfield_94MHz.txt', 'farfield_95MHz.txt', 'farfield_96MHz.txt', 'farfield_97MHz.txt', 'farfield_98MHz.txt', 'farfield_99MHz.txt', 'farfield_100MHz.txt', 'farfield_101MHz.txt', 'farfield_102MHz.txt', 'farfield_103MHz.txt', 'farfield_104MHz.txt', 'farfield_105MHz.txt', 'farfield_106MHz.txt', 'farfield_107MHz.txt', 'farfield_108MHz.txt', 'farfield_109MHz.txt', 'farfield_110MHz.txt', 'farfield_111MHz.txt', 'farfield_112MHz.txt', 'farfield_113MHz.txt', 'farfield_114MHz.txt', 'farfield_115MHz.txt', 'farfield_116MHz.txt', 'farfield_117MHz.txt', 'farfield_118MHz.txt', 'farfield_119MHz.txt', 'farfield_120MHz.txt', 'farfield_121MHz.txt', 'farfield_122MHz.txt', 'farfield_123MHz.txt', 'farfield_124MHz.txt', 'farfield_125MHz.txt', 'farfield_126MHz.txt', 'farfield_127MHz.txt', 'farfield_128MHz.txt', 'farfield_129MHz.txt', 'farfield_130MHz.txt', 'farfield_131MHz.txt', 'farfield_132MHz.txt', 'farfield_133MHz.txt', 'farfield_134MHz.txt', 'farfield_135MHz.txt', 'farfield_136MHz.txt', 'farfield_137MHz.txt', 'farfield_138MHz.txt', 'farfield_139MHz.txt', 'farfield_140MHz.txt', 'farfield_141MHz.txt', 'farfield_142MHz.txt', 'farfield_143MHz.txt', 'farfield_144MHz.txt', 'farfield_145MHz.txt', 'farfield_146MHz.txt', 'farfield_147MHz.txt', 'farfield_148MHz.txt', 'farfield_149MHz.txt', 'farfield_150MHz.txt', 'farfield_151MHz.txt', 'farfield_152MHz.txt', 'farfield_153MHz.txt', 'farfield_154MHz.txt', 'farfield_155MHz.txt', 'farfield_156MHz.txt', 'farfield_157MHz.txt', 'farfield_158MHz.txt', 'farfield_159MHz.txt', 'farfield_160MHz.txt', 'farfield_161MHz.txt', 'farfield_162MHz.txt', 'farfield_163MHz.txt', 'farfield_164MHz.txt', 'farfield_165MHz.txt', 'farfield_166MHz.txt', 'farfield_167MHz.txt', 'farfield_168MHz.txt', 'farfield_169MHz.txt', 'farfield_170MHz.txt', 'farfield_171MHz.txt', 'farfield_172MHz.txt', 'farfield_173MHz.txt', 'farfield_174MHz.txt', 'farfield_175MHz.txt', 'farfield_176MHz.txt', 'farfield_177MHz.txt', 'farfield_178MHz.txt', 'farfield_179MHz.txt', 'farfield_180MHz.txt', 'farfield_181MHz.txt', 'farfield_182MHz.txt', 'farfield_183MHz.txt', 'farfield_184MHz.txt', 'farfield_185MHz.txt', 'farfield_186MHz.txt', 'farfield_187MHz.txt', 'farfield_188MHz.txt', 'farfield_189MHz.txt', 'farfield_190MHz.txt', 'farfield_191MHz.txt', 'farfield_192MHz.txt', 'farfield_193MHz.txt', 'farfield_194MHz.txt', 'farfield_195MHz.txt', 'farfield_196MHz.txt', 'farfield_197MHz.txt', 'farfield_198MHz.txt', 'farfield_199MHz.txt', 'farfield_200MHz.txt', 'farfield_201MHz.txt', 'farfield_202MHz.txt', 'farfield_203MHz.txt', 'farfield_204MHz.txt', 'farfield_205MHz.txt', 'farfield_206MHz.txt', 'farfield_207MHz.txt', 'farfield_208MHz.txt', 'farfield_209MHz.txt', 'farfield_210MHz.txt', 'farfield_211MHz.txt', 'farfield_212MHz.txt', 'farfield_213MHz.txt', 'farfield_214MHz.txt', 'farfield_215MHz.txt', 'farfield_216MHz.txt', 'farfield_217MHz.txt', 'farfield_218MHz.txt', 'farfield_219MHz.txt', 'farfield_220MHz.txt', 'farfield_221MHz.txt', 'farfield_222MHz.txt', 'farfield_223MHz.txt', 'farfield_224MHz.txt', 'farfield_225MHz.txt', 'farfield_226MHz.txt', 'farfield_227MHz.txt', 'farfield_228MHz.txt', 'farfield_229MHz.txt', 'farfield_230MHz.txt', 'farfield_231MHz.txt', 'farfield_232MHz.txt', 'farfield_233MHz.txt', 'farfield_234MHz.txt', 'farfield_235MHz.txt', 'farfield_236MHz.txt', 'farfield_237MHz.txt', 'farfield_238MHz.txt', 'farfield_239MHz.txt', 'farfield_240MHz.txt', 'farfield_241MHz.txt', 'farfield_242MHz.txt', 'farfield_243MHz.txt', 'farfield_244MHz.txt', 'farfield_245MHz.txt', 'farfield_246MHz.txt', 'farfield_247MHz.txt', 'farfield_248MHz.txt', 'farfield_249MHz.txt', 'farfield_250MHz.txt']
    sim_beam_type: E-farfield
    feed_pol: x
    layout: 1 antenna
    port_num: 1
    ref_imp: 100
