// A visual proposal uses only pieces present in the demo. It never executes it.
const colors = { rojo: 'red', roja: 'red', red: 'red', azul: 'blue', blue: 'blue', verde: 'green', green: 'green', amarillo: 'yellow', amarilla: 'yellow', yellow: 'yellow' };
const labels = { red: 'roja', blue: 'azul', green: 'verde', yellow: 'amarilla' };
export function sceneProposal(plan, objects) {
  const empty = { targetIds: [], destination: null, note: '' };
  if (!plan || plan.status !== 'proposed' || plan.execution !== 'preview_only') return empty;
  const action = plan.actions.find(value => !value.negated && ['PICK_OBJECT', 'PLACE_OBJECT', 'MOVE_OBJECT', 'SORT_OBJECTS', 'INSPECT_OBJECT', 'INSPECT_BATCH', 'CLASSIFY_OBJECT'].includes(value.intent));
  if (!action) return empty;
  const parameters = action.parameters;
  const shapes = { pieza: null, bloque: 'cube', cubo: 'cube', cilindro: 'cylinder' };
  if (parameters.object_type && !(parameters.object_type in shapes)) return { ...empty, note: 'Ese tipo de objeto no está representado en la mesa.' };
  const shape = shapes[parameters.object_type];
  const color = colors[parameters.object_color];
  if (parameters.object_color && !color) return { ...empty, note: 'Ese color no está representado en la mesa.' };
  const feature = parameters.object_feature;
  if (feature && !['con perno', 'sin perno'].includes(feature)) return { ...empty, note: 'Esa característica no está representada en la mesa.' };
  if (parameters.object_quality) return { ...empty, note: 'Las piezas de ejemplo no tienen etiquetas de calidad medidas.' };
  const matching = objects.filter(object => (!color || object.color === color) && (!shape || object.shape === shape)
    && (!feature || (typeof object.with_pin === 'boolean' && object.with_pin === (feature === 'con perno'))));
  const quantity = Number.isInteger(parameters.quantity) ? parameters.quantity : null;
  const selected = quantity ? matching.slice(0, quantity) : matching;
  const targetIds = selected.map(object => object.id);
  if (!selected.length) return { ...empty, note: 'No hay piezas de ejemplo que coincidan con la orden.' };
  if (quantity && matching.length < quantity) return { targetIds, destination: null, note: `La mesa tiene ${matching.length} ${matching.length === 1 ? 'pieza coincidente' : 'piezas coincidentes'}; la orden pide ${quantity}. Ruta pendiente.` };
  const destinationColor = colors[parameters.destination_color];
  const destination = ['PLACE_OBJECT', 'MOVE_OBJECT'].includes(action.intent) && destinationColor
    ? { color: destinationColor, label: `Caja ${labels[destinationColor]} · propuesta` } : null;
  return { targetIds, destination, note: destination ? 'Ruta ilustrativa del plan · No ejecutada' : 'Selección propuesta sobre las piezas visibles' };
}
