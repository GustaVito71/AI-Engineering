"""SwitcherLLM: llama al mismo prompt con diferentes LLMs.

Patrones usados:
- Factory: crea el adaptador por nombre de proveedor.
- Strategy: el "current" objeto cambia el proveedor en tiempo de ejecución.
- Adapter: cada proveedor expone una interfaz común (chat()).

Este archivo es el CÓDIGO CLIENTE (y punto de entrada del programa):
solo conoce abstracciones (BaseProvider, ProviderFactory),
nunca los SDK concretos.

Punto de entrada:  python3 switcherllm.py
"""

import os

from providers import ProviderFactory, Message, LLMError, retry_with_backoff


class LLMClient:
    """Patrón Strategy: el proveedor actual es intercambiable en runtime.

    El "contexto" (esta clase) guarda un objeto provider. Ese objeto,
    la estrategia, se puede sustituir en caliente sin tocar este código.
    """

    def __init__(self, provider_name: str, **provider_kwargs):
        # La fábrica construye el adaptador; aquí solo almacenamos una referencia
        self.current = ProviderFactory.create(provider_name, **provider_kwargs)

    def switch(self, provider_name: str, **provider_kwargs) -> None:
        # Cambia la estrategia en caliente: se reemplaza el objeto provider
        self.current = ProviderFactory.create(provider_name, **provider_kwargs)

    def list_models(self) -> list[str]:
        # El cliente nunca sabe qué SDK hay detrás: solo conoce esta interface
        return self.current.get_available_models()

    @retry_with_backoff(max_retries=3, base_delay=1.0)  # reintentos ante rate limit
    def ask(self, prompt: str, **kwargs) -> str:
        message = [
            Message(role="system", content="Eres un asistente breve que responde en español."),
            Message(role="user", content=prompt),
        ]
        resp = self.current.chat(message, **kwargs)  # misma conversación, cualquier proveedor
        print(f"  {resp.content}\n")

        if resp.usage:
            print(f"  Tokens -> {resp.usage}\n")
        return resp.content


def main(client: LLMClient) -> None:
    print("Proveedores registrados en el factory:")
    for name in ProviderFactory.list_available():
        # Instanciar da acceso al modelo default sin llamar a la API
        try:
            provider = ProviderFactory.create(name)
            print(f"  - {name} [{provider.model}]")
        except LLMError as e:
            print(f"  - {name} [sin key]")  # sin API key configurada para ese proveedor
    print()

    available = ProviderFactory.list_available()

    # Bucle de la consola interactiva (REPL simple)
    while True:
        print(f"Proveedor actual: {client.current.name}")
        print("Proveedores disponibles: " + ", ".join(available))
        prompt = input("\nPrompt (o 'modelos', 'switch <proveedor>', 'salir'): ").strip()

        if prompt.lower() == "salir":
            break
        if prompt.lower() == "exit":
            break
        if prompt.lower() in ("modelos", "models", "list"):
            # Comando interno: NO hace falta llamar al LLM, es endpoint /models
            print(f"  Modelos de {client.current.name}: {client.list_models()}\n")
            continue
        if prompt.startswith("switch "):
            # Cambia la estrategia en tiempo de ejecución (patrón Strategy)
            target = prompt.split()[1].lower()
            if target in available:
                client.switch(target)
                print(f"-> Cambiado a {target}\n")
            else:
                print(f"Proveedor '{target}' no existe.\n")
            continue
        if not prompt:
            continue  # entrada vacía: vuelve a esperar

        # Entrada libre: la tratamos como un prompt para el LLM activo
        print(f"\n>>> Consultando a: {client.current.name}")
        try:
            client.ask(prompt)
        except LLMError as e:
            print(f"  ERROR: {e}\n")
        except Exception as e:
            print(f"  ERROR inesperado: {e}\n")


def run() -> None:
    """Punto de entrada real del programa.

    Se invoca tanto desde la consola (comando 'switcherllm' tras pip install)
    como al ejecutar 'python3 switcherllm.py'.
    """
    # Se lee la variable de entorno LLM_DEFAULT si existe, si no openai
    default_provider = os.environ.get("LLM_DEFAULT", "openai")
    client = LLMClient(default_provider)  # el cliente se crea UNA vez y se reutiliza
    print(f"\nIniciando SwitcherLLM con proveedor default: {default_provider}\n")
    main(client)


if __name__ == "__main__":
    # Este guard: el código solo corre si ejecutamos este script directamente,
    # no si se importa desde otro archivo.
    run()