import re


def _format_infogan_spec(args):
    cat_dim, cont_dim, bin_dim, noise_dim = args
    dims = [cat_dim, cont_dim, bin_dim, noise_dim]
    return ''.join(f"{dim}{key}" for dim, key in zip(dims, 'cgbn') if dim > 0)


def _parse_infogan_spec(spec):
    match = re.match(r"^(\d+c)?(\d+g)?(\d+b)?(\d+n)?$", spec)
    if match is None:
        raise ValueError(f"Invalid InfoGAN spec string: '{spec}'")
    groups = [match.group(i + 1) for i in range(4)]
    return tuple(0 if g is None else int(g[:-1]) for g in groups)


_PARSERS = {
    'InfoGAN': _parse_infogan_spec,
    'VAE': int,
    'GAN': int,
}
_FORMATTERS = {
    'InfoGAN': _format_infogan_spec,
    'VAE': str,
    'GAN': str,
}


def format_setup_spec(model_type, model_args, dataset_names):
    try:
        model_spec = _FORMATTERS[model_type](model_args)
    except KeyError:
        raise ValueError(f"Invalid model type: '{model_type}'. "
                         f"Expected one of {list(_FORMATTERS.keys())}")
    dataset_spec = '+'.join(dataset_names)
    return f"{model_type}-{model_spec}_{dataset_spec}"


def parse_setup_spec(string):
    match = re.match(r"^(.+)-(.+)_(.+)$", string)
    if match is None:
        raise ValueError(f"Invalid setup spec string: '{string}'")
    model_type, model_spec, dataset_spec = match.group(1), match.group(2), match.group(3)
    try:
        model_args = _PARSERS[model_type](model_spec)
    except KeyError:
        raise ValueError(f"Invalid model type: '{model_type}'. "
                         f"Expected one of {list(_PARSERS.keys())}")
    dataset_names = dataset_spec.split('+')
    return model_type, model_args, dataset_names


if __name__ == '__main__':
    for setup_spec in ["InfoGAN-10c2g0b62n_plain",
                       "InfoGAN-10c3g0b62n_plain+thin+thic",
                       "InfoGAN-X_plain+swel+frac",
                       "GANXX-10_plain+swel+frac",
                       "VAE-64_plain+thin+thic",
                       "GAN-64_plain"]:
        print(setup_spec)
        try:
            model_type, model_args, dataset_names = parse_setup_spec(setup_spec)
            print(model_type, model_args, dataset_names)
            print(format_setup_spec(model_type, model_args, dataset_names))
            print(format_setup_spec(model_type + 'a', model_args, dataset_names))
        except Exception as e:
            print(repr(e))
        print()
