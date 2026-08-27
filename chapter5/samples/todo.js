// Маленький файл на JavaScript — чтобы скобочный сканер Главы 5 было
// на чём увидеть. В курсе нет фронтенда, но индексатор обязан пережить
// чужой язык: в настоящем репозитории рядом с Python почти всегда лежит
// что-то ещё.
//
// Здесь нарочно собраны случаи, на которых наивный подсчёт скобок ломается:
// фигурная скобка внутри строки, скобка внутри комментария и вложенный
// объект внутри метода класса.

const STORAGE_KEY = "todo:items";

/**
 * Читает список задач из localStorage.
 * Возвращает пустой массив, если ничего не сохранено или JSON испорчен.
 */
function loadTodos() {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
        return [];
    }
    try {
        return JSON.parse(raw);
    } catch (error) {
        // Испорченный JSON — обычное дело: пользователь правил хранилище руками.
        console.warn("Не удалось прочитать задачи: {битый JSON}", error);
        return [];
    }
}

const formatTodo = (todo) => {
    // Скобка в строке ниже не открывает блок — сканер обязан это понять.
    const mark = todo.done ? "[x]" : "[ ]";
    return `${mark} ${todo.title}`;
};

class TodoList {
    constructor(items = []) {
        this.items = items;
    }

    add(title) {
        this.items.push({ title: title, done: false, meta: { source: "ui" } });
        return this.items.length;
    }

    toggle(index) {
        const todo = this.items[index];
        if (!todo) {
            return false;
        }
        todo.done = !todo.done;
        return true;
    }

    save() {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
    }
}

export { TodoList, loadTodos, formatTodo };
