# Real-ESRGAN photo fixture

`nasa-collins-896x1134.jpg` is a 1,016,064-pixel derivative of NASA's early
portrait of astronaut Eileen Collins. Credit: NASA. Public domain (US federal
work); used for restoration benchmarking and regression validation, without
implying endorsement.

- Source and public-domain statement: https://commons.wikimedia.org/wiki/File:Eileen_Collins,_early_NASA_portrait.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/f/f2/Eileen_Collins%2C_early_NASA_portrait.jpg
- NASA media guidance: https://www.nasa.gov/nasa-brand-center/images-and-media/
- Retrieved: 2026-10-26. Original dimensions: 2250 × 2848.
- Apple ImageIO (`sips --resampleHeightWidth 1134 896`) resized the photograph.
- SHA-256: `5956adbb2c9ac382e99aaa8be2890fa6c74fb09e0489c1e178cdbf8d701abdf7`

The source photograph contains a face, hair, woven clothing, insignia, small
lettering, and background gradients. It is a practical fixture, not an exhaustive
image-quality dataset. Keep this image identical across both models/precisions.
