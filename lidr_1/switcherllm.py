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

import sentry_sdk  # monitoreo de errores en la nube (Sentry.io)

from providers import ProviderFactory, Message, LLMError, retry_with_backoff
from providers.pricing import estimate_cost


class LLMClient:
    """Patrón Strategy: el proveedor actual es intercambiable en runtime.

    El "contexto" (esta clase) guarda un objeto provider. Ese objeto,
    la estrategia, se puede sustituir en caliente sin tocar este código.
    """

    def __init__(self, provider_name: str, system_prompt: str = None, **provider_kwargs):
        # La fábrica construye el adaptador; aquí solo almacenamos una referencia
        self.current = ProviderFactory.create(provider_name, **provider_kwargs)
        # Rol del asistente (system prompt): default o el pasado por el llamador
        self.system_prompt = system_prompt or "Eres un asistente breve que responde en español."

    def set_role(self, system_prompt: str) -> None:
        # Define el rol (system prompt) que verá el LLM antes del mensaje del usuario
        self.system_prompt = system_prompt

    def switch(self, provider_name: str, **provider_kwargs) -> None:
        # Cambia la estrategia en caliente: se reemplaza el objeto provider
        self.current = ProviderFactory.create(provider_name, **provider_kwargs)

    def list_models(self) -> list[str]:
        # El cliente nunca sabe qué SDK hay detrás: solo conoce esta interface
        return self.current.get_available_models()

    @retry_with_backoff(max_retries=3, base_delay=1.0)  # reintentos ante rate limit
    def ask(self, prompt: str, **kwargs) -> str:
        message = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        resp = self.current.chat(message, **kwargs)  # misma conversación, cualquier proveedor
        print(f"  {resp.content}\n")

        if resp.usage:
            print(f"  Tokens -> {resp.usage}")
        # El modelo que respondió lo trae la propia respuesta normalizada
        print(f"  Modelo -> {resp.model}")
        # estimate_cost devuelve (coste, nota): la nota explica la fuente o el motivo del fallo
        cost, cost_note = estimate_cost(resp.model, resp.usage)
        if cost is not None:
            print(f"  Coste estimado -> ${cost:.6f}\n")
        else:
            print(f"  Coste NO estimado -> {cost_note}\n")
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
    # Rol por defecto definido al iniciar (variable LLM_ROLE o su fallback)
    print(f"Rol por defecto: {client.system_prompt}")
    print()

    available = ProviderFactory.list_available()

    # Bucle de la consola interactiva (REPL simple)
    while True:
        print(f"Proveedor actual: {client.current.name}")
        print("Proveedores disponibles: " + ", ".join(available))
        prompt = input("\nPrompt (o 'modelos', 'switch <proveedor>', 'role <texto>', 'salir'): ").strip()

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
        if prompt.lower() in ("role", "rol", "system"):
            # Muestra el rol actual sin llamar al LLM
            print(f"  Rol actual: {client.system_prompt}\n")
            continue
        if prompt.lower().startswith(("role ", "rol ")):
            # Cambia el rol (system prompt) en caliente: afecta a la próxima consulta
            client.set_role(prompt.split(" ", 1)[1].strip())
            print(f"-> Rol actualizado: {client.system_prompt}\n")
            continue
        if not prompt:
            continue  # entrada vacía: vuelve a esperar

        # Entrada libre: la tratamos como un prompt para el LLM activo
        print(f"\n>>> Consultando a: {client.current.name}")
        try:
            client.ask(prompt)
        except LLMError as e:
            # Enviamos el error a Sentry (los LLMError también se reportan)
            sentry_sdk.capture_exception(e)
            print(f"  ERROR: {e}\n")
        except Exception as e:
            # Errores inesperados: a Sentry y se siguen mostrando en local
            sentry_sdk.capture_exception(e)
            print(f"  ERROR inesperado: {e}\n")


def run() -> None:
    """Punto de entrada real del programa.

    Se invoca tanto desde la consola (comando 'switcherllm' tras pip install)
    como al ejecutar 'python3 switcherllm.py'.
    """
    # Sentry: solo se activa si existe SENTRY_DSN en el entorno (.env)
    if dsn := os.environ.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=dsn, environment="development", traces_sample_rate=1.0)
        print("  -> Sentry habilitado: los errores se reportan a Sentry.io\n")

    # Se lee la variable de entorno LLM_DEFAULT si existe, si no anthropic
    default_provider = os.environ.get("LLM_DEFAULT", "anthropic")
    # Se lee la variable de entorno LLM_ROLE si existe, si no el rol por defecto
    default_role = os.environ.get("LLM_ROLE", "Eres un asistente breve que responde en español.")
    # el cliente se crea UNA vez y se reutiliza
    client = LLMClient(default_provider, system_prompt=default_role)
    print(f"\nIniciando SwitcherLLM con proveedor default: {default_provider}\n")
    main(client)


if __name__ == "__main__":
    # Este guard: el código solo corre si ejecutamos este script directamente,
    # no si se importa desde otro archivo.
    run()