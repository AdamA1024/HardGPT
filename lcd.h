#ifndef LCD_H
#define LCD_H

#include "ti_msp_dl_config.h"
#include <stdint.h>

#define LCD_I2C_ADDR 0x27

/* push a single byte out to the pcf8574 i2c expander */
static void pcf_send(uint8_t byte)
{
    while (!(DL_I2C_getControllerStatus(I2C_0_INST) &
             DL_I2C_CONTROLLER_STATUS_IDLE)) { }
    DL_I2C_fillControllerTXFIFO(I2C_0_INST, &byte, 1);
    DL_I2C_startControllerTransfer(I2C_0_INST, LCD_I2C_ADDR,
        DL_I2C_CONTROLLER_DIRECTION_TX, 1);
    while (DL_I2C_getControllerStatus(I2C_0_INST) &
           DL_I2C_CONTROLLER_STATUS_BUSY_BUS) { }
    volatile uint32_t d = 400;
    while (d--) { __asm__("nop"); }
}

/* toggle the enable line so the hd44780 latches whatever we just put on the bus */
static void lcd_pulse(uint8_t data)
{
    pcf_send(data | 0x04);
    volatile uint32_t d = 1000;
    while (d--) { __asm__("nop"); }
    pcf_send(data & ~0x04);
    d = 1000;
    while (d--) { __asm__("nop"); }
}

/* push 4 bits at a time. bit 0x08 keeps the backlight on. */
static void lcd_nibble(uint8_t nib, uint8_t rs)
{
    lcd_pulse((nib << 4) | 0x08 | rs);
}

/* the lcd takes a byte as two nibbles (high then low) once we're in 4-bit mode */
static void lcd_send_byte(uint8_t byte, uint8_t rs)
{
    lcd_nibble((byte >> 4) & 0x0F, rs);
    lcd_nibble(byte & 0x0F, rs);
}

static void lcd_command(uint8_t cmd)
{
    lcd_send_byte(cmd, 0);
    volatile uint32_t d = 80000;
    while (d--) { __asm__("nop"); }
}

static void lcd_putchar(char c)
{
    lcd_send_byte((uint8_t)c, 1);
    volatile uint32_t d = 5000;
    while (d--) { __asm__("nop"); }
}

static void lcd_puts(const char *s)
{
    while (*s) lcd_putchar(*s++);
}

static void lcd_set_cursor(uint8_t row, uint8_t col)
{
    uint8_t addr = (row == 0) ? col : (0x40 + col);
    lcd_command(0x80 | addr);
}

static void lcd_clear(void)
{
    lcd_command(0x01);
    volatile uint32_t d = 200000;
    while (d--) { __asm__("nop"); }
}

static void lcd_init(void)
{
    volatile uint32_t d;

    /* the datasheet says wait at least 40ms after power-up before talking */
    d = 2000000; while (d--) { __asm__("nop"); }

    /* hd44780 wake-up dance: 0x03 three times to force it into a known mode */
    lcd_nibble(0x03, 0);
    d = 200000; while (d--) { __asm__("nop"); }

    lcd_nibble(0x03, 0);
    d = 50000; while (d--) { __asm__("nop"); }

    lcd_nibble(0x03, 0);
    d = 50000; while (d--) { __asm__("nop"); }

    /* now flip it into 4-bit mode for the rest of the conversation */
    lcd_nibble(0x02, 0);
    d = 50000; while (d--) { __asm__("nop"); }

    lcd_command(0x28);  /* 4-bit interface, 2 lines, 5x8 font */
    lcd_command(0x0C);  /* display on, cursor + blink off */
    lcd_clear();
    lcd_command(0x06);  /* entry mode: cursor moves right, no display shift */
}

#endif
