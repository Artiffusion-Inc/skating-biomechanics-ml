import { z } from "zod"

export const RegisterRequestSchema = z.object({
  email: z.string().email("Введите корректный email"),
  password: z.string().min(8, "Минимум 8 символов").max(128),
  display_name: z.string().max(100).optional(),
})

export const LoginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

export const TokenResponseSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal("bearer"),
})

export const UserResponseSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  display_name: z.string().nullable(),
  avatar_url: z.string().nullable(),
  bio: z.string().nullable(),
  height_cm: z.number().int().nullable(),
  weight_kg: z.number().nullable(),
  language: z.string(),
  timezone: z.string(),
  theme: z.string(),
  is_active: z.boolean(),
  created_at: z.string(),
})

export const UpdateProfileRequestSchema = z.object({
  display_name: z.string().max(100).optional().nullable(),
  bio: z.string().optional().nullable(),
  height_cm: z.number().int().min(50).max(250).optional().nullable(),
  weight_kg: z.number().min(20).max(300).optional().nullable(),
})

export const UpdateSettingsRequestSchema = z.object({
  language: z.string().max(10).optional().nullable(),
  timezone: z.string().max(50).optional().nullable(),
  theme: z.enum(["light", "dark", "system"]).optional().nullable(),
})

export type RegisterRequest = z.infer<typeof RegisterRequestSchema>
export type LoginRequest = z.infer<typeof LoginRequestSchema>
export type TokenResponse = z.infer<typeof TokenResponseSchema>
export type UserResponse = z.infer<typeof UserResponseSchema>
export type UpdateProfileRequest = z.infer<typeof UpdateProfileRequestSchema>
export type UpdateSettingsRequest = z.infer<typeof UpdateSettingsRequestSchema>
