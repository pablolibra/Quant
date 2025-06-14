from nicegui import ui
import backend5  # Asegurate que esté presente y correcto

ui.add_head_html("""
<style>
    body {
        background-image: url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }
    .text-h6, .text-bold, .q-table, .q-table th, .q-table td, label, .q-notification {
        color: #ffffff !important;
        text-shadow: 0 0 5px rgba(0,0,0,0.7);
    }
    .q-table__top, .q-table__bottom {
        background: rgba(0, 0, 0, 0.5);
    }
    .q-table__container {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
</style>
""")

# === TÍTULO Y ESTILO ===
with ui.column().classes('items-center w-full'):
    ui.label('📈 Análisis de Portafolio Inteligente').classes('text-3xl font-bold text-primary')
    ui.label('Completá los datos y obtené tu diagnóstico financiero').classes('text-lg text-gray-500 mb-6')

# === INPUTS ===
with ui.card().classes('w-full max-w-xl mx-auto p-4'):
    tickers_input = ui.input(label='🧾 Tickers (separados por coma)').props('dense')
    pesos_input = ui.input(label='📊 Pesos en % (ej: 40,30,30)').props('dense')
    run_button = ui.button('🔍 Procesar', color='primary').classes('mt-4 w-full')

# === OUTPUTS ===
output_area = ui.column().classes('w-full max-w-5xl mx-auto mt-8')

# === FUNCIÓN DE EJECUCIÓN ===
def ejecutar():
    tickers_str = tickers_input.value
    pesos_str = pesos_input.value

    if not tickers_str or not pesos_str:
        ui.notify('Por favor completá ambos campos', type='warning')
        return

    try:
        resumen, fundamentales = backend5.run_analysis(tickers_str, pesos_str)
        output_area.clear()

        with output_area:
            ui.label('✅ Análisis completado').classes('text-green text-xl mb-2')

            ui.label('📄 Resumen del Portafolio').classes('text-h6 mt-4')
            ui.table(columns=[
                {'name': c, 'label': c, 'field': c} for c in resumen.columns
            ], rows=resumen.to_dict('records')).classes('w-full')

            ui.label('🧠 Señales Fundamentales').classes('text-h6 mt-6')
            for f in fundamentales:
                if f is None:
                    ui.label('⚠️ No se pudo obtener información de una empresa.').classes('text-red')
                else:
                    ui.label(f"{f['ticker']}: {f['Signal']} ({f['Score']}/6)").classes('text-bold mt-2')
                    for criterio, cumple in f['Criterios'].items():
                        icon = '✅' if cumple else '❌'
                        ui.label(f"{icon} {criterio}").classes('ml-2')

    except Exception as e:
        ui.notify(f"Error: {e}", type='negative')

run_button.on('click', ejecutar)

# === INICIAR LA APP ===

import os

port = int(os.environ.get("PORT", 10000))  # 10000 es el valor por defecto de Render
ui.run(host="0.0.0.0", port=port)