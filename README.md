# neural-style-transfer

Torch implementaition of http://arxiv.org/abs/1508.06576

I created this to serve as a baseline for my [fast-neural-style](https://github.com/ulucs/fast-neural-style/) code; but I'm pretty happy with how it turned out.

You can download VGG_16 from the [model zoo](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md); however it is not required as I've already included the top layers in this repo.

Loadcaffe is only required for slicing a perception model from scratch. Actual transfer does not need it.

Currently CPU-only