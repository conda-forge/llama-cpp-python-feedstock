import llama_cpp
print(llama_cpp.__version__)
print(llama_cpp.llama_backend_init())  # calls into libllama
llama_cpp.llama_backend_free()
