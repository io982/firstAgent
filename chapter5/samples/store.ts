// Файл на TypeScript — вторая половина корпуса для скобочного сканера.
// К случаям из todo.js добавлены типы и интерфейсы: в .ts верхнего уровня
// определений больше, и не все они содержат тело в фигурных скобках.

export interface Todo {
    title: string;
    done: boolean;
}

export type TodoFilter = "all" | "done" | "active";

/**
 * Хранилище задач с фильтрацией.
 * Обёртка над массивом: настоящая база задачам такого размера не нужна.
 */
export class TodoStore {
    private items: Todo[] = [];

    constructor(initial: Todo[] = []) {
        this.items = [...initial];
    }

    public add(title: string): number {
        return this.items.push({ title, done: false });
    }

    public filter(mode: TodoFilter): Todo[] {
        if (mode === "all") {
            return [...this.items];
        }
        return this.items.filter((todo) => (mode === "done" ? todo.done : !todo.done));
    }
}

export function countActive(store: TodoStore): number {
    return store.filter("active").length;
}
