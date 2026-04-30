export const skipAuth = process.env.NEXT_PUBLIC_SKIP_AUTH === "true"
export const devMockAuth = process.env.NEXT_PUBLIC_DEV_MOCK_AUTH === "true"
export const isDevelopment = process.env.NODE_ENV === "development"