# jaxtra

LAPACK ORMQR as a native JAX extension — apply Q from a QR factorisation to a matrix **without ever forming Q**.

Built on top of JAX's XLA Foreign Function Interface (FFI): a small C++/LAPACK kernel registered at runtime, no jaxlib rebuild required.
