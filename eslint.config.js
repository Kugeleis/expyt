export default [
    {
        ignores: ['**/node_modules/**', '**/.venv/**', '**/venv/**', '**/dist/**', '**/build/**']
    },
    {
        languageOptions: {
            ecmaVersion: 'latest',
            sourceType: 'module',
            globals: {
                window: 'readonly',
                document: 'readonly',
                fetch: 'readonly',
                console: 'readonly',
                setTimeout: 'readonly',
                clearTimeout: 'readonly',
                setInterval: 'readonly',
                clearInterval: 'readonly',
                alert: 'readonly',
                confirm: 'readonly',
                requestAnimationFrame: 'readonly',
                Chart: 'readonly',
                Error: 'readonly',
                HTMLElement: 'readonly',
                Event: 'readonly',
                CustomEvent: 'readonly',
                FormData: 'readonly',
                URL: 'readonly',
                FileReader: 'readonly',
                AbortController: 'readonly'
            }
        },
        rules: {
            'no-unused-vars': 'warn',
            'no-undef': 'error',
            semi: ['error', 'always'],
            quotes: ['error', 'single', { avoidEscape: true, allowTemplateLiterals: true }],
            'no-const-assign': 'error',
            'no-debugger': 'error',
            'no-dupe-args': 'error',
            'no-dupe-keys': 'error',
            'no-duplicate-case': 'error',
            'no-empty': ['error', { allowEmptyCatch: true }],
            'no-invalid-regexp': 'error',
            'no-unreachable': 'error',
            'use-isnan': 'error',
            'valid-typeof': 'error'
        }
    }
];
