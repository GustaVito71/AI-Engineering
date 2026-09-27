# Transcripción de reunión — Módulo de facturación (PagoYa)

Transcripción cruda de la reunión de kickoff. El **parámetro del ejercicio** es
el bloque de diálogo de abajo, delimitado por dos marcadores HTML (apertura y
cierre que dicen "transcripcion"). Es lo que se envía al endpoint
`POST /api/v1/estimate`. Está verificado que pasa la validación de entrada
(≥ 50 y ≤ 50.000 caracteres).

---

<!-- transcripcion -->
**Participantes:** Laura (producto), Martín (tech lead), Sofía (backend), Diego (frontend), cliente.

**Duración:** 45 min.

---

**Laura:** Buen día. Objetivo de hoy: definir alcance y límites del módulo de facturación para el MVP. Cliente, ¿qué es lo mínimo que necesitás salir a cobrar?

**Cliente:** Dos cosas: poder dar de alta clientes con su razón social y CUIT, y emitir comprobantes que cumplan con AFIP. Facturas, notas de crédito y recibos.

**Sofía:** Sobre AFIP: hoy la integración que tenemos no maneja el web service de comprobantes electrónicos. Habría que agregar el endpoint de facturación electrónica, con el certificado y las credenciales por cliente.

**Martín:** Cuidado con eso. El alta de AFIP es un punto de fricción: los tiempos del servicio externo no los controlamos nosotros. Propongo que para el MVP la emisión sea asincrónica: se encola el comprobante y se avisa cuando AFIP responde.

**Diego:** ¿Y el frontend? Necesito saber si va a haber un listado de comprobantes con su estado, porque si no, no puedo mostrar si una factura fue aceptada, rechazada o en proceso.

**Laura:** Sí, listado con filtros por tipo de comprobante y fecha, y una vista de detalle. También alta de clientes desde el panel. ¿Algo de reportes?

**Cliente:** Un reporte simple de cobranza: total facturado por mes. Con eso me alcanza.

**Sofía:** Para reportes necesito una capa de lectura, pero ya existe. Lo que sí falta es el registro de la factura como evento de negocio para poder agregar después otros reportes sin tocar el alta.

**Martín:** Tiempos. Backend: modelos de clientes y comprobantes, integración AFIP asincrónica con cola, reporte de cobranza, tests. Frontend: alta de clientes, listado con estados, detalle y reporte. Estimando con equipo chico, esto no baja de 80 horas en total, y con la incertidumbre de AFIP lo acompaña un buffer.

**Laura:** Limitemos el alcance entonces: MVP = alta de clientes, emisión asincrónica de facturas y notas de crédito, listado con estados y reporte de cobranza mensual. Queda afuera: recibos, módulo de cobros y alertas por email.

**Cliente:** De acuerdo. Para la entrega quiero poder emitir una factura contra un cliente real y verla en el listado.

**Martín:** Perfecto. Con ese alcance cerrado, armamos el desglose por tareas y lo presentamos con estimación en horas, equipo recomendado y duración en semanas. Pendientes de esta sesión: decidir si las notas de crédito comparten flujo con facturas (yo diría que sí) y definir políticas de reintento para la cola de AFIP.

**Laura:** Cierro entonces: alcance definido, pendientes anotados, y próxima sesión con la estimación desglosada.
<!-- /transcripcion -->