# SwitcherLLM

Mini aplicación CLI escrita en **Python** para entender cómo se comunica un programa con distintos LLM vía API key, desde el mismo código.

Manda el mismo prompt a **OpenAI**, **Anthropic (Claude)**, **Google Gemini** y **DeepSeek**, y cambia de proveedor en caliente. Diseñada con los patrones **Adapter**, **Factory** y **Strategy**, con manejo de errores y reintentos con backoff.

## Objetivo educativo

No busca ser una herramienta productiva, sino mostrar de forma mínima:

- La **estructura básica** de una llamada a la API de cada proveedor.
- Que cada API tiene **forma distinta** (Anthropic separa el system prompt, Gemini usa objetos `Content`, DeepSeek es compatible con OpenAI...).
- Cómo se manejan **credenciales** (`.env`, variables de entorno) de forma segura.
- Cómo se **normalizan errores** y se reintenta con backoff exponencial.

## Patrones usados

| Patrón | Dónde vive | Para qué |
|--------|------------|----------|
| **Adapter** | `providers/*_provider.py` | Traduce el SDK propio de cada proveedor a la interfaz común `BaseProvider` |
| **Factory** | `providers/factory.py` | Crea el adaptador correcto según el nombre, sin `if/elif` |
| **Strategy** | `switcherllm.py` (`LLMClient`) | El proveedor activo se intercambia en tiempo de ejecución |

```
┌──────────────┐   │   ┌─────────────────┐   ┌───────────────────────┐
│  switcherllm │──▶│   │   BaseProvider  │◀─▶│  OpenAIProvider       │
│  (Strategy)  │   │   │  (interfaz)     │   │  AnthropicProvider    │
│              │   │   └─────────────────┘   │  GeminiProvider       │
│  Factory     │──▶│        ▲   ▲   ▲        │  DeepSeekProvider     │
└──────────────┘   │        └───┼───┼────────┘  (Adapter pattern)    │
                   └────────────┼───┴────────────────────────────────┘
                                │  todos cumplen chat() y get_available_models()
```

## Estructura

```
lidr_1/
├── switcherllm.py           # Código cliente + REPL + punto de entrada
├── pyproject.toml           # Empaquetado y comando 'switcherllm'
├── requirements.txt         # Dependencias
├── .env.example             # Plantilla de credenciales (copia a .env)
└── providers/
    ├── __init__.py          # Carga el .env automáticamente
    ├── base.py              # Interfaz común: Message, LLMResponse, BaseProvider
    ├── errors.py            # Errores normalizados + retry_with_backoff
    ├── pricing.py           # Coste estimado por llamada (tabla local de LLMPrice)
    ├── factory.py           # ProviderFactory
    ├── openai_provider.py
    ├── anthropic_provider.py
    ├── gemini_provider.py
    └── deepseek_provider.py
```

## Requisitos

- Python 3.10+
- Una API key de al menos uno de los proveedores.

## Instalación

```bash
# 1. Entorno virtual
cd lidr_1
python3 -m venv venv_lidr1
source venv_lidr1/bin/activate

# 2. Dependencias (e instala el paquete + comando)
pip install -e .

# 3. Credenciales
cp .env.example .env   # y rellena con tus API keys
```

> También puedes correr sin instalar: `python3 switcherllm.py`

## Uso

```bash
source venv_lidr1/bin/activate
switcherllm            # o: python3 switcherllm.py
```

Dentro del REPL:

```
Prompt (o 'modelos', 'switch <proveedor>', 'role <texto>', 'salir'): switch gemini
-> Cambiado a gemini

Proveedor actual: gemini
Proveedores disponibles: openai, anthropic, gemini, deepseek

Prompt (o 'modelos', 'switch <proveedor>', 'role <texto>', 'salir'): modelos
  Modelos de gemini: ['gemini-3.6-flash', 'gemini-3.5-flash', ...]

Prompt (o 'modelos', 'switch <proveedor>', 'role <texto>', 'salir'): role Eres un crítico gastronómico severo.
-> Rol actualizado: Eres un crítico gastronómico severo.

Prompt (o 'modelos', 'switch <proveedor>', 'role <texto>', 'salir'): ¿Qué opinas de este menú?

| Comando | Acción |
|---------|--------|
| `switch <proveedor>` | Cambia de LLM en caliente (patrón Strategy) |
| `role <texto>` | Define el rol del asistente (system prompt) enviado antes de cada mensaje |
| `role` | Muestra el rol / system prompt actual |
| `modelos` | Lista los modelos disponibles del proveedor (consulta `/models` real) |
| texto libre | Se envía como prompt al proveedor activo |
| `salir` / `exit` | Cierra el programa |

El proveedor inicial se elige con la variable `LLM_DEFAULT` del `.env` (por defecto `anthropic`), y el rol inicial del asistente con `LLM_ROLE` (por defecto `Eres un asistente breve que responde en español.`). Ambos se muestran al iniciar y se pueden cambiar en caliente desde el REPL.

Proveedores soportados: `openai`, `anthropic`, `gemini`, `deepseek`.

## Seguridad de las API keys

- Las keys se almacenan en `.env` (permisos `600`), **nunca en el código**.
- `.gitignore` excluye `.env` del control de versiones.
- Cada adaptador lee su key del entorno con `os.environ.get(...)`.
- Precedencia: variable exportada en el shell > valor del `.env`.
- Si una key se filtra: revócarla en el dashboard del proveedor y generar otra.

## Coste estimado por llamada

- Tras cada consulta se muestran **tokens, modelo y coste estimado en USD**:

  ```
  Tokens -> {'input_tokens': 25, 'output_tokens': 35}
  Modelo -> claude-haiku-4-5-20251001
  Coste estimado -> $0.000043
  ```

- Se calcula con `providers/pricing.py` usando la librería **LLMPrice** (base local de precios por millón de tokens, sin llamadas de red). Fórmula: `(input×precio_in + output×precio_out) / 1.000.000`.
- Los tokens de todos los proveedores se **normalizan** a `input_tokens` / `output_tokens` en cada adaptador, para que el cálculo no dependa del SDK.
- Si el modelo no está en la base de precios (versión muy reciente o snapshot desactualizado), se muestra un **coste NO estimado** con el motivo, sin romper la consulta:

  ```
  Tokens -> {'input_tokens': 12, 'output_tokens': 9}
  Modelo -> gemini-3.6-flash
  Coste NO estimado -> LLMPrice no tiene datos para 'gemini-3.6-flash' en el snapshot 2026.4.3 (de abril)
  ```

- La fecha del snapshot se lee automáticamente de la versión instalada de `llmprice-kit` (el paquete versiona sus datos por fecha: `2026.4.3` = 3 de abril de 2026).

## Manejo de errores y reintentos

- Todos los fallos se envuelven en `LLMError` (o subtipos `AuthenticationError`, `RateLimitError`), para que el código cliente no dependa del SDK concreto.
- El decorador `retry_with_backoff` reintenta **solo** los errores recuperables (`RateLimitError`) con espera exponencial (1s, 2s, 4s...).
- Errores de autenticación **no** se reintentan: no tiene sentido, hay que corregir la credencial.