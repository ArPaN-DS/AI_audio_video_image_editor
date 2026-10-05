const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
    constructor(id) {
        this.id = id;
        this.attributes = {};
        this.listeners = {};
        this.inert = false;
        this.tabIndex = 0;
        this.classes = new Set();
        this.classList = {
            contains: name => this.classes.has(name),
            add: name => this.classes.add(name),
            remove: name => this.classes.delete(name)
        };
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    querySelector() { return null; }
    querySelectorAll() { return []; }
    contains(target) { return this === target; }
    focus() { document.activeElement = this; }
    click() { this.listeners.click({ target: this }); }
}

const root = new Element('root');
root.setAttribute('data-theme', 'dark');
const themeButton = new Element('themeToggle');
const menuButton = new Element('mobileMenuBtn');
const closeButton = new Element('closeMobileNav');
const drawer = new Element('mobileNavDrawer');
const backdrop = new Element('mobileNavBackdrop');
const navigationLink = new Element('last-link');
const main = new Element('main');
drawer.classes.add('hidden');
drawer.querySelectorAll = () => [closeButton, navigationLink];
drawer.contains = target => [drawer, closeButton, navigationLink].includes(target);
const elements = [themeButton, menuButton, closeButton, drawer, backdrop];
const listeners = {};
const document = {
    readyState: 'complete',
    documentElement: root,
    activeElement: menuButton,
    body: { style: { overflow: 'auto' }, children: [main, drawer] },
    getElementById: id => elements.find(element => element.id === id),
    addEventListener: (name, callback) => { listeners[name] = callback; }
};
const mediaQueries = new Map();
const windowListeners = {};
const storage = new Map();
const sandbox = {
    document,
    window: {
        dispatchEvent() {},
        addEventListener(name, callback) { windowListeners[name] = callback; },
        matchMedia(query) {
            const mediaQuery = { addEventListener(name, callback) { this.listener = callback; } };
            mediaQueries.set(query, mediaQuery);
            return mediaQuery;
        }
    },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    CustomEvent: class { constructor(name, options) { this.name = name; this.detail = options.detail; } }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/theme.js', 'utf8'), sandbox);
assert.equal(themeButton.getAttribute('aria-pressed'), 'true');
themeButton.click();
assert.equal(root.getAttribute('data-theme'), 'light');
assert.equal(storage.get('theme'), 'light');
menuButton.click();
assert.equal(main.inert, true);
assert.equal(document.activeElement, closeButton);
assert.equal(drawer.getAttribute('aria-hidden'), 'false');
assert.equal(document.body.style.overflow, 'hidden');
navigationLink.focus();
listeners.keydown({ key: 'Tab', shiftKey: false, preventDefault() {}, stopPropagation() {} });
assert.equal(document.activeElement, closeButton);
listeners.keydown({ key: 'Tab', shiftKey: true, preventDefault() {}, stopPropagation() {} });
assert.equal(document.activeElement, navigationLink);
listeners.keydown({ key: 'Escape', preventDefault() {}, stopPropagation() {} });
assert.equal(document.activeElement, menuButton);
assert.equal(main.inert, false);
assert.equal(document.body.style.overflow, 'auto');
assert.equal(drawer.getAttribute('aria-hidden'), 'true');
main.inert = true;
menuButton.click();
mediaQueries.get('(min-width: 769px)').listener({ matches: true });
assert.equal(main.inert, true);
assert.equal(menuButton.getAttribute('aria-expanded'), 'false');
windowListeners.storage({ key: 'theme', newValue: 'dark' });
assert.equal(root.getAttribute('data-theme'), 'dark');
assert.equal(themeButton.getAttribute('aria-pressed'), 'true');
console.log('Theme and keyboard navigation regression checks passed.');
