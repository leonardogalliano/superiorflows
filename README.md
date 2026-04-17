# Superiorflows

[![CI](https://github.com/leonardogalliano/superiorflows/actions/workflows/ci.yml/badge.svg)](https://github.com/leonardogalliano/superiorflows/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/leonardogalliano/superiorflows/branch/main/graph/badge.svg)](https://codecov.io/gh/leonardogalliano/superiorflows)

**Superiorflows** is a JAX-based library for sampling physical systems using continuous normalising flows.

The project is developed at the Centre des Sciences des Données at the École Normale Supérieure (Paris), and is largely inspired by the PyTorch implementation [learndiffeq](https://github.com/h2o64/learndiffeq).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for dependency management.

To install the project and its dependencies:

```bash
uv sync
```

For GPU support (CUDA 12):

```bash
uv sync --extra cuda
```

## Design principles

**Superiorflows** is designed to operate on *arbitrary structured inputs*, ranging from arrays of any rank to complex pytrees representing physical systems.

While a primary focus is on particle-based systems, the core abstractions are intentionally fully general and not tied to a specific domain. The library provides flexible building blocks rather than fixed pipelines.

Users are expected to define:
- custom input structures (e.g. particle systems, graphs, fields),
- velocity / vector field architectures,
- and data pipelines tailored to their problem.

This design enables the same framework to be applied across a wide range of sampling tasks beyond its original scope.

## Ecosystem

The library builds on the JAX ecosystem, in particular:
- [**Equinox**](https://github.com/patrick-kidger/equinox) for model definition and PyTree-based modules
- [**Diffrax**](https://github.com/patrick-kidger/diffrax) for continuous-time integration
- [**Distrax**](https://github.com/google-deepmind/distrax) for probabilistic components
- [**Optax**](https://github.com/google-deepmind/optax) for training
- [**Orbax**](https://github.com/google/orbax) for checkpointing
- [**Grain**](https://github.com/google/grain) for data pipelines

## Development

The project is currently under active development, and the API may change frequently.

## Disclaimer

Large Language Models (LLMs) were used for code checks, documentation, and test writing. Project contributors remain fully responsible for the codebase and its correctness.